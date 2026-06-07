"""
node_airllm.py — Узел Noumind: Reverse WebSocket клиент + AirLLM inference.

Архитектура (Reverse WS):
  Узел сам подключается к Gateway WS-серверу (порт 8003).
  Это решает проблему NAT — Gateway не нужно знать IP узла.

Алгоритм:
  1. Подключаемся к ws://gateway:8003
  2. Отправляем {"type":"register", ...} — Gateway назначает слои
  3. Получаем {"type":"registered", "layer_start":..., "layer_end":...}
  4. Загружаем модель через AirLLM
  5. Слушаем задачи {"type":"generate", ...}, отвечаем {"type":"result", ...}
  6. При разрыве — переподключаемся через 5 сек

NOTE: AirLLM НЕ поддерживает layer_start/layer_end.
      Слои виртуальные (tracking покрытия). Модель полная ~0.4 GB RAM.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']  = 'TRUE'
os.environ['OMP_NUM_THREADS']        = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
import argparse
import json
import time
import socket
import torch
import websockets

# ─────────────────────────────────────────
# АРГУМЕНТЫ
# ─────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Noumind node — подключается к Gateway через Reverse WebSocket"
)
parser.add_argument('--worker-id',
    default=f'user-{socket.gethostname()}',
    help='Уникальный ID этого узла')
parser.add_argument('--gateway-ws',
    default='ws://217.160.49.222:8003',
    help='WebSocket адрес Gateway (порт 8003)')
parser.add_argument('--model',
    default='mistralai/Mistral-7B-Instruct-v0.2',
    help='HuggingFace модель')
parser.add_argument('--layer-start', type=int, default=-1,
    help='Принудительный старт слоёв (-1 = от Gateway)')
parser.add_argument('--layer-end',   type=int, default=-1,
    help='Принудительный конец слоёв (-1 = от Gateway)')
args = parser.parse_args()

WORKER_ID  = args.worker_id
GATEWAY_WS = args.gateway_ws

# ─────────────────────────────────────────
# УСТРОЙСТВО
# ─────────────────────────────────────────
if torch.cuda.is_available():
    device  = 'cuda'
    tier    = 'gpu'
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    ram_gb  = vram_gb
elif torch.backends.mps.is_available():
    device  = 'mps'
    tier    = 'fast_cpu'
    import psutil
    ram_gb  = psutil.virtual_memory().available / 1024**3
    vram_gb = 0.0
else:
    device  = 'cpu'
    tier    = 'slow_cpu'
    import psutil
    ram_gb  = psutil.virtual_memory().available / 1024**3
    vram_gb = 0.0

print(f"[Node] Worker : {WORKER_ID}")
print(f"[Node] Gateway: {GATEWAY_WS}")
print(f"[Node] Device : {device} ({tier})")
print(f"[Node] RAM    : {ram_gb:.1f} GB | VRAM: {vram_gb:.1f} GB")

# ─────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful multilingual AI assistant.
Always respond in the same language as the user's question.
If the user writes in Russian, respond in Russian.
Be concise and helpful. Do not repeat the question."""

# ─────────────────────────────────────────
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ─────────────────────────────────────────
model_loaded       = None
layer_start_global = 0
layer_end_global   = 31

# ─────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ
# ─────────────────────────────────────────
def load_model(layer_start: int, layer_end: int):
    """
    Загружаем AirLLM.
    NOTE: AirLLM НЕ поддерживает layer_start/layer_end — грузим всю модель,
    но layer streaming держит RAM ~0.4 GB независимо от числа слоёв.
    """
    from airllm import AutoModel

    count  = layer_end - layer_start + 1
    gb_est = count * 0.44
    print(f"[Node] Загружаю модель... "
          f"(виртуальные слои {layer_start}-{layer_end}, ~{gb_est:.1f} GB весов)")

    mdl = AutoModel.from_pretrained(
        args.model,
        device      = "cpu" if device in ("cpu", "mps") else device,
        dtype       = torch.float16,
        max_seq_len = 512,
    )
    print(f"[Node] ✓ Модель загружена ({args.model})")
    return mdl

# ─────────────────────────────────────────
# INFERENCE — greedy decode
# ─────────────────────────────────────────
def run_inference(mdl, message: str, max_new_tokens: int = 300) -> str:
    """
    Ручной greedy decode через AirLLM (layer streaming с диска).
    NOTE: AirLLM не поддерживает .generate() — делаем вручную.
    ~15-30 с/токен на CPU.
    """
    prompt    = f"[INST] {SYSTEM_PROMPT}\n\n{message} [/INST]"
    input_ids = mdl.tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=512
    )
    generated  = input_ids.clone()
    eos_id     = mdl.tokenizer.eos_token_id or 2
    prompt_len = input_ids.shape[1]

    print(f"[Node] Промпт: {prompt_len} токенов → генерирую до {max_new_tokens}...")

    with torch.no_grad():
        for step in range(max_new_tokens):
            t0     = time.time()
            out    = mdl(generated)
            logits = out.logits if hasattr(out, "logits") else out[0]
            next_id = int(logits[:, -1, :].argmax(-1).item())
            tok     = mdl.tokenizer.decode([next_id], skip_special_tokens=True)
            elapsed = time.time() - t0

            print(f"[Node] [{step+1:3d}/{max_new_tokens}] '{tok}' ({elapsed:.1f}s)",
                  flush=True)

            if next_id == eos_id:
                print(f"[Node] EOS на шаге {step + 1}")
                break

            generated = torch.cat(
                [generated, torch.tensor([[next_id]])], dim=1
            )

    new_ids = generated[0][prompt_len:].tolist()
    text    = mdl.tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    return text

# ─────────────────────────────────────────
# ОСНОВНОЙ ЦИКЛ — подключение и работа
# ─────────────────────────────────────────
async def connect_and_work():
    """
    Подключаемся к Gateway WebSocket.
    Регистрируемся, загружаем модель, слушаем задачи.
    При разрыве переподключаемся через 5 сек.
    """
    global model_loaded, layer_start_global, layer_end_global

    while True:
        try:
            print(f"[Node] Подключаюсь к {GATEWAY_WS}...", flush=True)

            async with websockets.connect(
                GATEWAY_WS,
                ping_interval = 20,
                ping_timeout  = 60,
                close_timeout = 10,
                open_timeout  = 15,
            ) as ws:

                # ── 1. Регистрация ────────────────────────
                reg_msg = {
                    "type":        "register",
                    "worker_id":   WORKER_ID,
                    "tier":        tier,
                    "device":      device,
                    "ram_gb":      ram_gb,
                    "vram_gb":     vram_gb,
                    "model":       args.model,
                    "layer_start": args.layer_start,
                    "layer_end":   args.layer_end,
                }
                await ws.send(json.dumps(reg_msg))

                # ── 2. Подтверждение Gateway ──────────────
                raw  = await asyncio.wait_for(ws.recv(), timeout=30)
                data = json.loads(raw)

                if data.get("type") != "registered":
                    print(f"[Node] Неожиданный ответ: {data}")
                    continue

                layer_start_global = data["layer_start"]
                layer_end_global   = data["layer_end"]
                cov = data.get("coverage", {})

                print(f"[Node] ✓ Зарегистрирован!")
                print(f"[Node] Слои: {layer_start_global}-{layer_end_global} "
                      f"({layer_end_global - layer_start_global + 1} шт)")
                print(f"[Node] Покрытие сети: {cov.get('percent','?')}% "
                      f"({cov.get('covered','?')}/{cov.get('total','?')})")

                # ── 3. Загрузка модели (один раз) ─────────
                if model_loaded is None:
                    print(f"[Node] Загружаю модель...", flush=True)
                    loop = asyncio.get_event_loop()
                    model_loaded = await loop.run_in_executor(
                        None, load_model, layer_start_global, layer_end_global
                    )
                    # Сообщаем Gateway что готовы принимать задачи
                    await ws.send(json.dumps({
                        "type":      "ready",
                        "worker_id": WORKER_ID,
                    }))
                else:
                    print(f"[Node] Модель уже загружена, сразу готов")
                    await ws.send(json.dumps({
                        "type":      "ready",
                        "worker_id": WORKER_ID,
                    }))

                print(f"[Node] Слушаю задачи...\n", flush=True)

                # ── 4. Основной цикл задач ────────────────
                async for message in ws:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "generate":
                        task_id = data["task_id"]
                        text    = data.get("message", "")

                        print(f"\n[Node] Задача {task_id[:8]}: '{text[:50]}'")
                        t0 = time.time()

                        try:
                            loop = asyncio.get_event_loop()
                            result_text = await loop.run_in_executor(
                                None, run_inference, model_loaded, text
                            )
                            elapsed = time.time() - t0
                            print(f"[Node] ✓ Готово за {elapsed:.1f}с: "
                                  f"'{result_text[:60]}'")

                            await ws.send(json.dumps({
                                "type":      "result",
                                "task_id":   task_id,
                                "text":      result_text,
                                "elapsed":   round(elapsed, 2),
                                "worker_id": WORKER_ID,
                                "layers":    f"{layer_start_global}-{layer_end_global}",
                            }))

                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            await ws.send(json.dumps({
                                "type":      "result",
                                "task_id":   task_id,
                                "text":      f"Ошибка: {e}",
                                "elapsed":   0,
                                "worker_id": WORKER_ID,
                            }))

                    elif msg_type == "pong":
                        pass  # ответ на heartbeat

                    elif msg_type == "ping":
                        await ws.send(json.dumps({
                            "type":      "pong",
                            "worker_id": WORKER_ID,
                        }))

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[Node] Соединение закрыто: {e}")
        except OSError as e:
            print(f"[Node] Сетевая ошибка: {e}")
        except Exception as e:
            print(f"[Node] Ошибка: {e}")

        print(f"[Node] Переподключаюсь через 5 сек...", flush=True)
        await asyncio.sleep(5)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
async def main():
    await connect_and_work()


if __name__ == "__main__":
    asyncio.run(main())
