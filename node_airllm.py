# -*- coding: utf-8 -*-
"""
Узел Noumind — распределённый pipeline.
CUDA:  AirLLM native (layer_start / layer_end, streaming с диска)
Mac:   transformers + ручное выделение слоёв + MPS
CPU:   то же, device=cpu
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import asyncio
import argparse
import torch
import httpx

# ─────────────────────────────────────────
# АРГУМЕНТЫ
# ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--gateway',     default='http://217.160.49.222:8002')
parser.add_argument('--worker-id',   default=f'user-{os.uname().nodename}')
parser.add_argument('--layer-start', type=int, default=0)
parser.add_argument('--layer-end',   type=int, default=15)
parser.add_argument('--model',       default='mistralai/Mistral-7B-Instruct-v0.2')
parser.add_argument('--force-cpu',   action='store_true')
args = parser.parse_args()

GATEWAY_URL = args.gateway
WORKER_ID   = args.worker_id
LAYER_START = args.layer_start
LAYER_END   = args.layer_end
MODEL_NAME  = args.model

# ─────────────────────────────────────────
# ОПРЕДЕЛЯЕМ УСТРОЙСТВО И МЕТОД ЗАГРУЗКИ
# ─────────────────────────────────────────
def detect_device():
    if args.force_cpu:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

DEVICE = detect_device()
USE_AIRLLM_CUDA = (DEVICE == 'cuda')

tier_label = 'gpu' if DEVICE == 'cuda' else ('fast_cpu' if DEVICE == 'mps' else 'slow_cpu')

print(f"[Node] Worker     : {WORKER_ID}")
print(f"[Node] Gateway    : {GATEWAY_URL}")
print(f"[Node] Model      : {MODEL_NAME}")
print(f"[Node] Слои       : {LAYER_START}-{LAYER_END}")
print(f"[Node] Device     : {DEVICE}")
print(f"[Node] Метод      : {'AirLLM CUDA' if USE_AIRLLM_CUDA else 'Transformers + ' + DEVICE.upper()}")
print()

# ─────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ
# ─────────────────────────────────────────
model_layers = None   # torch.nn.ModuleList — только наши слои
embed_in     = None   # embedding (только у первого узла, LAYER_START == 0)
lm_head      = None   # lm_head (только у последнего узла)
norm_final   = None   # final norm (только у последнего узла)
is_first     = (LAYER_START == 0)
is_last      = None   # определяется после загрузки конфига

def load_model_transformers():
    """Загружаем полную модель, оставляем только нужные слои, остальное удаляем."""
    global model_layers, embed_in, lm_head, norm_final, is_last
    from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
    import gc

    print(f"[Node] Загружаю конфиг {MODEL_NAME}...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    total_layers = config.num_hidden_layers
    is_last = (LAYER_END >= total_layers - 1)

    print(f"[Node] Всего слоёв в модели: {total_layers}")
    print(f"[Node] is_first={is_first}, is_last={is_last}")
    print(f"[Node] Загружаю веса... (может занять несколько минут)")

    # Загружаем в bfloat16 для экономии памяти
    dtype = torch.bfloat16
    full_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map='cpu',        # сначала на CPU — потом переносим нужное
        low_cpu_mem_usage=True,
    )

    print(f"[Node] Веса загружены, выделяю слои {LAYER_START}-{LAYER_END}...")

    # Извлекаем только наши слои
    all_layers   = full_model.model.layers
    model_layers = torch.nn.ModuleList(
        [all_layers[i] for i in range(LAYER_START, min(LAYER_END + 1, len(all_layers)))]
    ).to(DEVICE)

    if is_first:
        embed_in = full_model.model.embed_tokens.to(DEVICE)
        print(f"[Node] Embedding загружен (первый узел)")

    if is_last:
        norm_final = full_model.model.norm.to(DEVICE)
        lm_head    = full_model.lm_head.to(DEVICE)
        print(f"[Node] lm_head + norm загружены (последний узел)")

    # Освобождаем остальную модель
    del full_model
    import gc; gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()

    params = sum(p.numel() for m in model_layers for p in m.parameters()) // 1_000_000
    print(f"[Node] Слои {LAYER_START}-{LAYER_END} на {DEVICE} | {params}M параметров")


def load_model_airllm():
    """CUDA: используем AirLLM с нативным layer_start/layer_end."""
    global model_layers, is_last
    from airllm import AutoModel
    from transformers import AutoConfig

    config   = AutoConfig.from_pretrained(MODEL_NAME)
    is_last  = (LAYER_END >= config.num_hidden_layers - 1)

    print(f"[Node] AirLLM: загружаю слои {LAYER_START}-{LAYER_END} с диска...")
    model_layers = AutoModel.from_pretrained(
        MODEL_NAME,
        layer_start=LAYER_START,
        layer_end=LAYER_END,
    )
    print(f"[Node] AirLLM: слои {LAYER_START}-{LAYER_END} загружены (CUDA)")


print(f"[Node] Загружаю модель...")
if USE_AIRLLM_CUDA:
    load_model_airllm()
else:
    load_model_transformers()

print(f"[Node] Готов к обработке запросов")
print()

# ─────────────────────────────────────────
# ОБРАБОТКА АКТИВАЦИЙ
# ─────────────────────────────────────────
def process_sync(activations_bytes: bytes) -> bytes:
    buf   = io.BytesIO(activations_bytes)
    hidden = torch.load(buf, map_location=DEVICE, weights_only=False)

    with torch.no_grad():
        if USE_AIRLLM_CUDA:
            # AirLLM обрабатывает через свой интерфейс
            out = model_layers(hidden)
        else:
            # transformers: прогоняем через наши слои последовательно
            if is_first and embed_in is not None:
                # hidden сюда приходит как input_ids (Long tensor)
                hidden = embed_in(hidden.to(DEVICE))

            # Каждый decoder layer принимает (hidden_states,) → (hidden_states, ...)
            for layer in model_layers:
                layer_out = layer(hidden)
                hidden    = layer_out[0]  # берём только hidden_states

            if is_last:
                hidden = norm_final(hidden)
                logits = lm_head(hidden)
                out    = logits
            else:
                out = hidden

    buf_out = io.BytesIO()
    # Переносим на CPU перед сериализацией (MPS → CPU для передачи)
    torch.save(out.cpu(), buf_out)
    return buf_out.getvalue()


# ─────────────────────────────────────────
# РЕГИСТРАЦИЯ
# ─────────────────────────────────────────
async def register(client: httpx.AsyncClient):
    try:
        await client.post(
            f"{GATEWAY_URL}/register",
            json={
                "worker_id":   WORKER_ID,
                "layer_start": LAYER_START,
                "layer_end":   LAYER_END,
                "model":       MODEL_NAME,
                "is_fallback": False,
                "tier":        tier_label,
            },
            timeout=10,
        )
        print(f"[Node] Зарегистрирован → {GATEWAY_URL}")
    except Exception as e:
        print(f"[Node] Gateway недоступен ({e}) — работаем, ждём подключения")


# ─────────────────────────────────────────
# WORKER LOOP
# ─────────────────────────────────────────
async def worker_loop():
    async with httpx.AsyncClient(timeout=120) as client:
        await register(client)

        print(f"[Node] Worker loop запущен — слои {LAYER_START}-{LAYER_END} ждут задач")

        while True:
            try:
                resp = await client.get(
                    f"{GATEWAY_URL}/pipeline/task",
                    params={
                        "worker_id":   WORKER_ID,
                        "layer_start": LAYER_START,
                        "layer_end":   LAYER_END,
                    },
                )
                if resp.status_code != 200:
                    await asyncio.sleep(2)
                    continue

                task = resp.json()
                if not task.get("task_id"):
                    await asyncio.sleep(1)
                    continue

                task_id = task["task_id"]
                print(f"[Node] Задача {task_id[:8]} | слои {LAYER_START}-{LAYER_END}")

                act_resp = await client.get(
                    f"{GATEWAY_URL}/pipeline/activations/{task_id}"
                )
                act_bytes = act_resp.content
                print(f"[Node] Активации: {len(act_bytes):,} байт")

                loop        = asyncio.get_event_loop()
                output_bytes = await loop.run_in_executor(None, process_sync, act_bytes)

                await client.post(
                    f"{GATEWAY_URL}/pipeline/submit",
                    params={
                        "task_id":   task_id,
                        "worker_id": WORKER_ID,
                        "layer_end": LAYER_END,
                    },
                    content=output_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                )
                print(f"[Node] Готово {task_id[:8]} → {len(output_bytes):,} байт отправлено")

            except Exception as e:
                print(f"[Node] Ошибка: {e}")
                await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(worker_loop())
