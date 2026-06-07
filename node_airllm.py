"""
node_airllm.py — Узел Noumind: Reverse WebSocket клиент + selective layer loading.

Архитектура (Reverse WS):
  Узел сам подключается к Gateway WS-серверу (порт 8003).
  Это решает проблему NAT — Gateway не нужно знать IP узла.

Алгоритм:
  1. Подключаемся к ws://gateway:8003
  2. Отправляем {"type":"register", ...} — Gateway назначает слои
  3. Получаем {"type":"registered", "layer_start":..., "layer_end":...}
  4. Скачиваем только нужный safetensors-шард (~4.5 GB вместо ~14 GB)
  5. Загружаем только назначенные слои в память
  6. Слушаем задачи {"type":"generate", ...}, отвечаем {"type":"result", ...}
  7. При разрыве — переподключаемся через 5 сек

Слои выровнены по файлам Mistral-7B:
  model-00001 → слои  0-10  (~4.5 GB)
  model-00002 → слои 11-21  (~4.5 GB)
  model-00003 → слои 22-31  (~4.5 GB)
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
import httpx
from huggingface_hub import hf_hub_download

# EFCT — adaptive gates
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuron import Neuron

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
parser.add_argument('--gateway-http', default='http://217.160.49.222:8002',
    help='HTTP адрес Gateway для EFCT федерации (порт 8002)')
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
# EFCT — adaptive gates
# ─────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GATEWAY_HTTP  = args.gateway_http
GATES_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"gates_{socket.gethostname()}.pt")
FEDERATION_N  = 10   # федеративная синхронизация каждые N задач

# Загружаем сохранённые gates если есть, иначе — новый нейрон
if os.path.exists(GATES_PATH):
    efct = Neuron.load(GATES_PATH)
    print(f"[EFCT] Gates загружены: {GATES_PATH} (задач: {efct.tasks_done})")
else:
    efct = Neuron()
    print(f"[EFCT] Новый нейрон инициализирован")

tasks_done_efct = efct.tasks_done   # продолжаем счётчик с сохранённого

# ─────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful multilingual AI assistant.
Always respond in the same language as the user's question.
If the user writes in Russian, respond in Russian.
Be concise and helpful. Do not repeat the question."""

# ─────────────────────────────────────────
# МАППИНГ СЛОЁВ → ФАЙЛЫ (Mistral-7B-Instruct-v0.2)
# ─────────────────────────────────────────

# Каждый файл содержит ≈11 decoder-слоёв (~4.5 GB).
# Узел скачивает ТОЛЬКО файл(ы) для своих слоёв.
LAYER_TO_FILE = {
    (0,  10): "model-00001-of-00003.safetensors",
    (11, 21): "model-00002-of-00003.safetensors",
    (22, 31): "model-00003-of-00003.safetensors",
}

# ─────────────────────────────────────────
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ─────────────────────────────────────────
model_loaded       = None   # dict{"model": ..., "tokenizer": ...}
layer_start_global = 0
layer_end_global   = 31


def get_my_files(layer_start: int, layer_end: int) -> list:
    """Возвращает список safetensors-файлов для указанного диапазона слоёв."""
    needed = []
    for (fs, fe), fname in LAYER_TO_FILE.items():
        # Пересечение диапазонов
        if layer_start <= fe and layer_end >= fs:
            needed.append(fname)
    return needed


# ─────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ — только нужные слои
# ─────────────────────────────────────────

def load_model(layer_start: int, layer_end: int) -> dict:
    """
    1. Скачиваем конфиг-файлы (маленькие) + только нужный safetensors-шард.
    2. Инициализируем модель из конфига (без весов, очень быстро).
    3. Загружаем веса нашего шарда (strict=False — остальные слои игнорируются).
    4. Оставляем только layer_start..layer_end, остальные удаляем из памяти.

    Итог: в RAM только наши слои (~1.5 GB для 11 слоёв fp16).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from safetensors.torch import load_file as safetensors_load
    import gc

    model_id = args.model
    files    = get_my_files(layer_start, layer_end)

    print(f"[Node] Слои {layer_start}-{layer_end} → файлы: {files}")

    # ── 1. Конфиг и токенайзер (маленькие файлы) ─────────────────────────
    config_files = [
        "config.json", "tokenizer.json", "tokenizer_model",
        "tokenizer_config.json", "special_tokens_map.json",
        "generation_config.json",
    ]
    for cfg in config_files:
        try:
            hf_hub_download(repo_id=model_id, filename=cfg)
        except Exception:
            pass  # некоторые файлы могут отсутствовать

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"[Node] ✓ Токенайзер загружен")

    # ── 2. Скачиваем только нужные шарды ─────────────────────────────────
    shard_paths = []
    for fname in files:
        print(f"[Node] Скачиваю {fname}...", flush=True)
        path = hf_hub_download(repo_id=model_id, filename=fname)
        shard_paths.append(path)
        print(f"[Node] ✓ {fname} → {path}")

    # ── 3. Инициализируем структуру модели без весов ──────────────────────
    print(f"[Node] Создаю структуру модели (без весов)...")
    config = AutoConfig.from_pretrained(model_id)
    model  = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    # ── 4. Загружаем веса нашего шарда (остальные слои — случайные, неважно) ──
    for path in shard_paths:
        state = safetensors_load(path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Node] ✓ Веса загружены | пропущено: {len(missing)} ключей")

    # ── 5. Оставляем только наши слои ────────────────────────────────────
    all_layers = list(model.model.layers)
    model.model.layers = torch.nn.ModuleList(all_layers[layer_start:layer_end + 1])
    model.eval()

    # Освобождаем память
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    count = layer_end - layer_start + 1
    print(f"[Node] ✓ Модель готова: {count} слоёв ({layer_start}-{layer_end}) | "
          f"файлов: {len(shard_paths)}")

    return {"model": model, "tokenizer": tokenizer}


# ─────────────────────────────────────────
# INFERENCE — forward через назначенные слои
# ─────────────────────────────────────────

def run_inference(mdl_bundle: dict, message: str, max_new_tokens: int = 200) -> str:
    """
    Greedy decode через частичную модель (только наши слои).
    Полноценный ответ — когда узел покрывает все 32 слоя (слои 0-31).
    Для частичных узлов: возвращает скрытые состояния / промежуточный вывод.

    ~5-15 с/токен на CPU.
    """
    model     = mdl_bundle["model"]
    tokenizer = mdl_bundle["tokenizer"]

    prompt    = f"[INST] {SYSTEM_PROMPT}\n\n{message} [/INST]"
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=512
    )
    generated  = input_ids.clone()
    eos_id     = tokenizer.eos_token_id or 2
    prompt_len = input_ids.shape[1]

    print(f"[Node] Промпт: {prompt_len} токенов → генерирую до {max_new_tokens}...")

    with torch.no_grad():
        for step in range(max_new_tokens):
            t0     = time.time()
            out    = model(generated)
            logits = out.logits if hasattr(out, "logits") else out[0]
            next_id = int(logits[:, -1, :].argmax(-1).item())
            tok     = tokenizer.decode([next_id], skip_special_tokens=True)
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
    text    = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    return text

# ─────────────────────────────────────────
# EFCT — внешний сигнал качества через Groq
# ─────────────────────────────────────────

async def get_quality_signal(message: str, answer: str) -> float | None:
    """
    Groq Llama-3.1-8b-instant оценивает качество ответа → [0.0, 1.0].
    Возвращает None если GROQ_API_KEY не задан или запрос упал.
    Timeout 8с — не блокирует pipeline при недоступности Groq.
    """
    if not GROQ_API_KEY or not answer.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system",
                         "content": "Rate answer quality 0.0-1.0. Return ONLY a number."},
                        {"role": "user",
                         "content": f"Question: {message}\nAnswer: {answer}\nScore:"},
                    ],
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
            )
        score = float(resp.json()["choices"][0]["message"]["content"].strip())
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"[EFCT] Groq недоступен: {e}")
        return None


async def federate_gates():
    """
    Отправляем текущие gates на Gateway (/gates/upload),
    скачиваем глобальный агрегат (/gates/global) и применяем.
    """
    global efct
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Upload
            await client.post(
                f"{GATEWAY_HTTP}/gates/upload",
                params={"worker_id": WORKER_ID, "tasks_done": efct.tasks_done},
                json={"gates": efct.gates.cpu().tolist(),
                      "phase": efct.phase.cpu().tolist()},
            )
            # Download global aggregate
            resp = await client.get(f"{GATEWAY_HTTP}/gates/global")
            data = resp.json()
            if data.get("status") == "ok" and data.get("gates"):
                efct.gates = torch.tensor(data["gates"], dtype=efct.gates.dtype)
                efct.phase = torch.tensor(data["phase"], dtype=efct.phase.dtype)
                print(f"[EFCT] Федерация применена "
                      f"(участников: {data.get('contributors', '?')})", flush=True)
    except Exception as e:
        print(f"[EFCT] Федерация недоступна: {e}", flush=True)


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
                    print(f"[Node] Скачиваю и загружаю слои "
                          f"{layer_start_global}-{layer_end_global}...", flush=True)
                    loop = asyncio.get_event_loop()
                    model_loaded = await loop.run_in_executor(
                        None, load_model, layer_start_global, layer_end_global
                    )
                else:
                    print(f"[Node] Модель уже загружена, сразу готов")

                # Сообщаем Gateway что готовы принимать задачи
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

                            # ── EFCT: reward сигнал → обновление gates ───────
                            global tasks_done_efct
                            quality = await get_quality_signal(text, result_text)
                            if quality is not None:
                                efct.local_update(quality)
                                signal_src = "groq"
                            else:
                                efct.local_update(0.5)   # нейтральный fallback
                                signal_src = "neutral"

                            tasks_done_efct += 1
                            efct.save(GATES_PATH)
                            print(f"[EFCT] task={tasks_done_efct} "
                                  f"signal={signal_src}({quality or 0.5:.3f}) "
                                  f"baseline={efct.baseline:.4f} "
                                  f"dist={efct.identity_distance():.6f}", flush=True)

                            if tasks_done_efct % FEDERATION_N == 0:
                                await federate_gates()
                            # ─────────────────────────────────────────────────

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
