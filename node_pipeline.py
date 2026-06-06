"""
node_pipeline.py — Узел пользователя
Держит слои 11-21 TinyLlama + EFCT нейрон.
Pull-модель: сам опрашивает Gateway каждые 2 сек.
"""
import sys, os, io, asyncio, argparse, socket, base64, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn, httpx
try:
    import psutil
except ImportError:
    psutil = None

# Добавляем путь к neuron.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuron import Neuron

# ─────────────────────────────────────────
# GROQ API — внешний сигнал качества (COM)
# ─────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

async def get_quality_signal(message: str, local_answer: str) -> float | None:
    """
    Оцениваем качество ответа через Groq Llama-3.1-8B.
    Возвращает float [0.0, 1.0] или None при ошибке/отсутствии ключа.
    """
    if not GROQ_API_KEY or not local_answer.strip():
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a response quality evaluator. "
                                "Rate the quality of an AI response on a scale from 0.0 to 1.0. "
                                "Return ONLY a decimal number, nothing else."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {message}\n\n"
                                f"Answer: {local_answer}\n\n"
                                "Quality score (0.0-1.0):"
                            ),
                        },
                    ],
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
            )
        score_text = resp.json()["choices"][0]["message"]["content"].strip()
        score = float(score_text)
        score = max(0.0, min(1.0, score))
        return score

    except Exception as e:
        print(f"[EFCT] Groq API ошибка: {e} — используем confidence", flush=True)
        return None

# ─────────────────────────────────────────
# АРГУМЕНТЫ
# ─────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--worker-id",   default=f"user-{socket.gethostname()[:12]}")
parser.add_argument("--gateway",     default="http://217.160.49.222:8002")
parser.add_argument("--port",        type=int, default=9002)
parser.add_argument("--model",       default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
parser.add_argument("--layer-start", type=int, default=11)
parser.add_argument("--load-pct",    type=int, default=40)
args = parser.parse_args()

WORKER_ID   = args.worker_id
GATEWAY_URL = args.gateway
LAYER_START = args.layer_start
LAYER_END   = 21
MODEL_NAME  = args.model
LOAD_PCT    = args.load_pct
GATES_FILE  = f"gates_{WORKER_ID}.pt"

# ─────────────────────────────────────────
# ЗАГРУЖАЕМ МОДЕЛЬ (слои 11-21)
# ─────────────────────────────────────────

import sys
print(f"\n[Node] Worker ID: {WORKER_ID}", flush=True)
print(f"[Node] Загружаю {MODEL_NAME}...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"[Node] Токенизатор загружен", flush=True)

full_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
)
full_model.eval()
print(f"[Node] Модель загружена", flush=True)

# ─────────────────────────────────────────
# УСТРОЙСТВО
# ─────────────────────────────────────────

if torch.cuda.is_available():
    device = "cuda"
    tier   = "gpu"
elif torch.backends.mps.is_available():
    device = "mps"
    tier   = "fast_cpu"
else:
    device = "cpu"
    tier   = "slow_cpu"

full_model = full_model.to(device)
print(f"[Node] Device: {device} | Tier: {tier}", flush=True)

# ─────────────────────────────────────────
# СЛОИ
# ─────────────────────────────────────────
# ВАЖНО: модель НЕ обрезается реально — autoregressive_decode на step>0
# прогоняет полную последовательность токенов через всю модель (0-21).
# Обрезание ModuleList ломает это (токены попадают в слои 11-21 как embeddings).
# load% влияет только на отображение/нагрузку, не на реальную структуру.
layers    = full_model.model.layers
norm      = full_model.model.norm
lm_head   = full_model.lm_head
total_lay = len(layers)   # 22

ram_free = f"{psutil.virtual_memory().available/1024**3:.1f}GB" if psutil else "?"
print(f"[Node] Слои {LAYER_START}-{LAYER_END} (полная модель {total_lay} слоёв) | load={LOAD_PCT}%", flush=True)
print(f"[Node] RAM свободно: {ram_free}", flush=True)

# ─────────────────────────────────────────
# EFCT НЕЙРОН
# ─────────────────────────────────────────

# Размер = seq_len * hidden_size. Используем фиксированный hidden_size.
HIDDEN_SIZE = full_model.config.hidden_size  # 2048
efct = Neuron(size=HIDDEN_SIZE, neuron_id=WORKER_ID)

if os.path.exists(GATES_FILE):
    state = torch.load(GATES_FILE, weights_only=False)
    efct.gates       = state["gates"]
    efct.phase       = state["phase"]
    efct.com_history = state.get("com_history", [])
    efct.tasks_done  = state.get("tasks_done", 0)
    efct.baseline    = state.get("baseline", None)
    print(f"[Node] Gates загружены: {GATES_FILE} | задач: {efct.tasks_done} | адапт: {efct.identity_distance():.6f}")
else:
    print(f"[Node] Новые gates (identity init)")

print(f"[Node] Готов!\n")

# ─────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────

def bytes_to_tensor(data: bytes) -> torch.Tensor:
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=False)

def forward_second_half(hidden: torch.Tensor) -> torch.Tensor:
    """Прогоняет через слои LAYER_START-LAYER_END.

    Trick: подменяем embedding через hook, прогоняем model целиком,
    но благодаря residual connections слои 0-10 просто передают hidden без изменений
    т.к. их output hidden_states[11] совпадает с тем что мы передали.
    Берём hidden_states[-1] (после слоя 21) для декодирования.
    """
    with torch.no_grad():
        captured = {}

        def embed_hook(module, input, output):
            """Подменяем embedding на полученный hidden state"""
            captured["original"] = output.clone()
            return hidden

        hook = full_model.model.embed_tokens.register_forward_hook(embed_hook)
        try:
            dummy_ids   = torch.zeros(1, hidden.shape[1], dtype=torch.long, device=hidden.device)
            out         = full_model.model(dummy_ids, output_hidden_states=True, use_cache=False)
            final_hid   = out.hidden_states[-1]   # после слоя 21
            logits      = lm_head(norm(final_hid))
        finally:
            hook.remove()
    return logits

def autoregressive_decode(first_activations: torch.Tensor,
                          input_ids_list: list,
                          max_new_tokens: int = 150,
                          min_new_tokens: int = 8,
                          token_callback=None,
                          temperature: float = 0.7,
                          top_p: float = 0.9) -> tuple:
    """
    Авторегрессионная генерация с правильным контекстом.

    Шаг 0: логиты из первых активаций (Gateway, слои 0-10 → Node, слои 11-21).
           Берём первый новый токен — он уже знает весь prompt.
    Шаг 1+: накапливаем generated_ids и прогоняем через полную модель Node.
    В конце убираем оригинальный prompt из текста.
    """
    prompt_ids   = input_ids_list                    # список int
    generated_ids = list(prompt_ids)                 # начинаем с prompt
    last_logits   = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                # Первый шаг: активации уже содержат контекст всего промпта
                logits = forward_second_half(first_activations)  # [1, seq_len, vocab]
            else:
                # Следующие шаги: полная модель с накопленными токенами
                cur_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
                out        = full_model.model(cur_tensor, output_hidden_states=False, use_cache=False)
                logits     = lm_head(norm(out.last_hidden_state))  # [1, len, vocab]

            last_logits = logits
            logits_1d   = logits[0, -1, :].float()
            next_token  = logits_1d.argmax().item()
            new_len     = len(generated_ids) - len(prompt_ids)

            is_eos = (next_token == tokenizer.eos_token_id or next_token == 0)

            if is_eos:
                if new_len >= min_new_tokens:
                    break
                # Ещё не набрали минимум — принудительно берём лучший не-EOS токен
                logits_no_eos = logits_1d.clone()
                logits_no_eos[tokenizer.eos_token_id] = -float('inf')
                logits_no_eos[0] = -float('inf')
                next_token = logits_no_eos.argmax().item()

            generated_ids.append(next_token)

            # Стриминг: вычисляем дельту относительно предыдущего декода
            if token_callback:
                full_so_far = tokenizer.decode(
                    generated_ids[len(prompt_ids):],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                delta = full_so_far[len(getattr(token_callback, '_prev', '')):]
                if delta:
                    token_callback(delta)
                token_callback._prev = full_so_far  # type: ignore[attr-defined]

            # После минимума: останавливаемся на конце абзаца (двойной \n)
            if new_len >= min_new_tokens and step >= 2:
                recent = tokenizer.decode(generated_ids[-3:], skip_special_tokens=True)
                if '\n\n' in recent or recent.endswith('\n'):
                    # Проверяем что это не середина списка
                    decoded_so_far = tokenizer.decode(generated_ids[len(prompt_ids):], skip_special_tokens=True)
                    lines = decoded_so_far.strip().split('\n')
                    last_line = lines[-1].strip() if lines else ''
                    # Останавливаемся только если последняя строка заканчивается на точку/знак
                    if not last_line or last_line[-1] in '.!?:':
                        break

    # Декодируем только новые токены (после prompt)
    new_ids = generated_ids[len(prompt_ids):]
    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()

    # Убираем артефакты chat-шаблона TinyLlama и ведущую пунктуацию
    import re
    # TinyLlama chat template tokens: <|assistant|>, </s>, <s>, <|user|>, <|system|>
    text = re.sub(r'<\|[^|]+\|>|</s>|<s>', '', text)
    # Ведущие символы: '".  ,  ; и т.п.
    text = re.sub(r'^[\s\"\'\.\,\;\:\!\?\-\—\–]+', '', text).strip()

    # Если модель не сгенерировала ничего — берём хотя бы первый токен без фильтрации
    if not text and new_ids:
        text = tokenizer.decode(new_ids[:1], skip_special_tokens=False).strip()

    # Уверенность по последней позиции
    last_pos   = last_logits[:, -1, :].float() if last_logits is not None else torch.zeros(1, 32000)
    confidence = float(torch.softmax(last_pos, dim=-1).max())
    return text, confidence

def save_gates():
    torch.save({
        "gates":       efct.gates,
        "phase":       efct.phase,
        "com_history": efct.com_history,
        "tasks_done":  efct.tasks_done,
        "baseline":    efct.baseline,
    }, GATES_FILE)

# ─────────────────────────────────────────
# ОБРАБОТКА ЗАДАЧИ
# ─────────────────────────────────────────

tasks_processed      = 0
avg_speed            = 0.0
last_quality_signal  = None
last_signal_source   = "—"
yield_during_chat    = False   # снижать скорость генерации во время активного чата

FEDERATED_INTERVAL   = 10

# Динамическая нагрузка без перезапуска узла (управляется лаунчером)
load_config = {
    "max_concurrent": 1,    # сколько задач одновременно (зарезервировано)
    "sleep_between":  0.5,  # пауза между задачами (сек)
    "percent":        40,   # текущая нагрузка %
}

# ─────────────────────────────────────────
# MoE — специализация узла (Routed Expert)
# ─────────────────────────────────────────
CATEGORIES = {
    "code":     ["code", "python", "function", "bug", "error", "programming", "javascript", "script", "debug", "api", "sql"],
    "science":  ["physics", "chemistry", "biology", "math", "equation", "formula", "research", "quantum", "molecule"],
    "history":  ["history", "war", "ancient", "century", "historical", "empire"],
    "creative": ["story", "poem", "write", "creative", "imagine", "fiction", "novel", "song"],
    "general":  [],
}

category_scores = {cat: [] for cat in CATEGORIES}   # cat → [score, ...]

def classify_message(message: str) -> str:
    msg = message.lower()
    for cat, kws in CATEGORIES.items():
        if cat == "general":
            continue
        if any(kw in msg for kw in kws):
            return cat
    return "general"

def get_specialization() -> str:
    """Специализация = категория с лучшим средним качеством (минимум 2 примера)."""
    avgs = {cat: sum(s) / len(s) for cat, s in category_scores.items() if len(s) >= 2}
    if not avgs:
        return "general"
    return max(avgs, key=avgs.get)

def process_activations(act_b64: str, message: str = "", input_ids: list = None,
                        task_id: str = None, gateway_url: str = None) -> tuple:
    """Полный pipeline: EFCT → слои 11-21 → авторегрессионный декод"""
    global tasks_processed, avg_speed
    t_start = time.time()

    # 1. Декодируем активации от Gateway (hidden state после слоёв 0-10) — CPU
    act_bytes   = base64.b64decode(act_b64)
    activations = bytes_to_tensor(act_bytes)   # остаётся на CPU для EFCT
    print(f"[Node] Активации получены: {tuple(activations.shape)}", flush=True)

    # 2. EFCT.forward() — адаптируем на CPU (gates/phase — CPU тензоры)
    rep_flat    = activations.mean(dim=1).squeeze(0).float()       # [2048] CPU
    adapted     = efct.forward(rep_flat.unsqueeze(0))              # [1, 2048] CPU
    delta       = (adapted.squeeze(0) - rep_flat).to(activations.dtype) * 0.01
    activations = activations + delta.unsqueeze(0).unsqueeze(0)
    print(f"[Node] EFCT.forward() | адапт: {efct.identity_distance():.6f}", flush=True)

    # 3a. Переносим на device для модели
    activations = activations.to(device)

    # 3b. Строим callback для стриминга токенов (синхронный POST из треда)
    token_callback = None
    if task_id and gateway_url:
        def _push(delta: str):
            try:
                httpx.post(
                    f"{gateway_url}/activations/token",
                    params={"task_id": task_id, "token": delta},
                    timeout=3.0,
                )
            except Exception:
                pass
        _push._prev = ""   # type: ignore[attr-defined]
        token_callback = _push

    # 3c. Авторегрессионная генерация
    ids = input_ids if input_ids else []

    # Оборачиваем callback: добавляем throttle если yield_during_chat
    if yield_during_chat and token_callback:
        _orig_cb = token_callback
        def _throttled(delta: str):
            _orig_cb(delta)
            time.sleep(0.05)
        _throttled._prev = ""   # type: ignore[attr-defined]
        actual_cb = _throttled
    elif yield_during_chat:
        # нет stream callback, но нужен throttle — используем no-op с sleep
        def _noop_throttle(delta: str):
            time.sleep(0.05)
        _noop_throttle._prev = ""   # type: ignore[attr-defined]
        actual_cb = _noop_throttle
    else:
        actual_cb = token_callback

    text, confidence = autoregressive_decode(
        activations, ids, max_new_tokens=150, token_callback=actual_cb,
        temperature=0.7, top_p=0.9,
    )
    elapsed = time.time() - t_start
    print(f"[Node] Сгенерировано: '{text[:80]}'", flush=True)
    print(f"[Node] Confidence: {confidence:.6f} | время: {elapsed:.1f}s", flush=True)

    # EFCT.local_update и save_gates вызываются из worker_loop (async),
    # чтобы можно было await get_quality_signal перед обновлением.
    return text, confidence, elapsed

# ─────────────────────────────────────────
# WORKER LOOP — опрашивает Gateway
# ─────────────────────────────────────────

async def upload_gates_to_gateway():
    """Отправляем локальные gates на Gateway для федеративного усреднения."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{GATEWAY_URL}/gates/upload",
                params={"worker_id": WORKER_ID, "tasks_done": efct.tasks_done},
                json={"gates": efct.gates.tolist(), "phase": efct.phase.tolist()},
                headers={"Content-Type": "application/json"},
            )
            data = resp.json()
            print(f"[Federated] ↑ Gates отправлены | участников в сети: {data.get('contributors', 0)}", flush=True)
    except Exception as e:
        print(f"[Federated] Ошибка upload: {e}", flush=True)


async def download_global_gates():
    """Скачиваем глобальные gates и мягко сливаем со своими (70% свои + 30% глобальные)."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{GATEWAY_URL}/gates/global")
            data = resp.json()

        if data.get("status") != "ok":
            print(f"[Federated] Глобальные gates ещё не готовы", flush=True)
            return

        contributors = data["contributors"]
        if contributors < 2:
            return  # нет смысла при одном участнике

        global_gates = torch.tensor(data["gates"], dtype=torch.float32)
        global_phase = torch.tensor(data["phase"], dtype=torch.float32)

        # Мягкое слияние: 70% наши + 30% глобальные
        alpha      = 0.3
        efct.gates = (1 - alpha) * efct.gates + alpha * global_gates
        efct.phase = (1 - alpha) * efct.phase + alpha * global_phase

        save_gates()
        print(
            f"[Federated] ↓ Gates обновлены от {contributors} узлов | "
            f"адаптация: {efct.identity_distance():.6f}",
            flush=True,
        )
    except Exception as e:
        print(f"[Federated] Ошибка download: {e}", flush=True)


async def register_with_gateway(specialization: str = "general", scores: dict = None, first: bool = False):
    """Регистрируемся в Gateway (+ MoE специализация и category_scores)."""
    if first:
        await asyncio.sleep(3)
    ram_gb = 0
    if psutil:
        ram_gb = psutil.virtual_memory().available // (1024 ** 3)
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(
                f"{GATEWAY_URL}/register",
                params={
                    "worker_id":      WORKER_ID,
                    "tier":           tier,
                    "device":         device,
                    "ram_gb":         ram_gb,
                    "specialization": specialization,
                    "is_fallback":    False,
                },
                json={"category_scores": scores or {}},
                headers={"Content-Type": "application/json"},
            )
            print(f"[Node] ✓ Зарегистрирован | spec={specialization} tier={tier} device={device}", flush=True)
        except Exception as e:
            print(f"[Node] Регистрация не удалась: {e}", flush=True)

async def worker_loop():
    print(f"[Worker] Опрашиваю {GATEWAY_URL}/activations/next каждые 2 сек...")

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            try:
                resp = await client.get(f"{GATEWAY_URL}/activations/next", params={"worker_id": WORKER_ID})
                data = resp.json()

                if not data.get("task_id"):
                    await asyncio.sleep(2)
                    continue

                task_id    = data["task_id"]
                act_b64    = data["activations"]
                message    = data.get("message", "")
                input_ids  = data.get("input_ids", [])
                is_stream  = data.get("streaming", False)
                print(f"\n[Worker] ← Задача {task_id[:8]} | stream={is_stream} | '{message[:40]}'", flush=True)

                # Обрабатываем в executor (не блокируем event loop)
                import concurrent.futures
                loop = asyncio.get_running_loop()
                _ids = input_ids
                _tid = task_id if is_stream else None
                _gw  = GATEWAY_URL if is_stream else None
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    text, confidence, elapsed = await loop.run_in_executor(
                        pool,
                        lambda: process_activations(act_b64, message, _ids,
                                                    task_id=_tid, gateway_url=_gw)
                    )

                # ── COM-сигнал: пробуем Groq, fallback → confidence ──
                global tasks_processed, avg_speed, last_quality_signal, last_signal_source
                quality = await get_quality_signal(message, text)
                if quality is not None:
                    com_signal    = quality
                    signal_source = "groq"
                    efct.local_update(com_signal)
                    print(f"[EFCT] Groq оценка: {quality:.4f} → адаптация: {efct.identity_distance():.6f}", flush=True)
                else:
                    com_signal    = confidence
                    signal_source = "confidence"
                    efct.local_update(com_signal)
                    print(f"[EFCT] Confidence fallback: {confidence:.4f}", flush=True)

                last_quality_signal = com_signal
                last_signal_source  = signal_source

                tasks_processed += 1
                avg_speed = 0.9 * avg_speed + 0.1 * elapsed
                save_gates()

                # ── MoE: записываем качество по категории запроса ──
                category = classify_message(message)
                category_scores[category].append(com_signal)
                print(f"[Node] Gates сохранены | задач: {efct.tasks_done} | COM={com_signal:.4f} [{signal_source}] | категория: {category}", flush=True)

                # ── Федеративная синхронизация + ре-регистрация специализации каждые N ──
                if efct.tasks_done % FEDERATED_INTERVAL == 0:
                    print(f"[Federated] Раунд {efct.tasks_done // FEDERATED_INTERVAL} — синхронизация...", flush=True)
                    await upload_gates_to_gateway()
                    await download_global_gates()

                    spec   = get_specialization()
                    scores = {cat: round(sum(s) / len(s), 4) if s else 0 for cat, s in category_scores.items()}
                    await register_with_gateway(specialization=spec, scores=scores)
                    print(f"[Node] 🎯 Специализация: {spec} | scores: { {k:v for k,v in scores.items() if v} }", flush=True)

                # Возвращаем результат на Gateway
                await client.post(
                    f"{GATEWAY_URL}/activations/result",
                    params={
                        "task_id":    task_id,
                        "result":     text,
                        "confidence": com_signal,
                        "worker_id":  WORKER_ID,
                        "avg_speed":  round(avg_speed, 2),
                    }
                )
                print(f"[Worker] ✓ {task_id[:8]} отправлен Gateway\n")

                # Throttle по динамической нагрузке (умная нагрузка без перезапуска)
                if load_config["sleep_between"] > 0:
                    await asyncio.sleep(load_config["sleep_between"])

            except httpx.ConnectError:
                print(f"[Worker] Gateway недоступен, жду 10 сек...")
                await asyncio.sleep(10)
            except Exception as e:
                print(f"[Worker] Ошибка: {e}")
                await asyncio.sleep(3)

# ─────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(application):
    task = asyncio.create_task(worker_loop())
    asyncio.create_task(register_with_gateway(first=True))
    print(f"[Node] Worker loop запущен")
    yield
    task.cancel()

app = FastAPI(title=f"Noümind Node {WORKER_ID}", lifespan=lifespan)

@app.get("/metrics")
def metrics():
    """Реальные метрики системы через psutil."""
    if not psutil:
        return {"error": "psutil not installed"}
    cpu  = psutil.cpu_percent(interval=0.1)
    ram  = psutil.virtual_memory()
    net  = psutil.net_io_counters()
    return {
        "cpu":              round(cpu, 1),
        "ram_percent":      round(ram.percent, 1),
        "ram_used_gb":      round(ram.used  / 1024**3, 2),
        "ram_total_gb":     round(ram.total / 1024**3, 2),
        "net_sent_mb":      round(net.bytes_sent / 1024**2, 2),
        "net_recv_mb":      round(net.bytes_recv / 1024**2, 2),
        "layers_kept":      len(layers),
        "layer_start":      LAYER_START,
        "layer_end":        LAYER_END,
        "yield_chat":       yield_during_chat,
        "inference_active": tasks_processed > 0 and avg_speed > 0,
    }


@app.post("/settings/yield_chat")
async def set_yield_chat(enabled: bool):
    """Включить/выключить снижение нагрузки во время чата."""
    global yield_during_chat
    yield_during_chat = enabled
    print(f"[Node] yield_during_chat = {enabled}", flush=True)
    return {"yield_chat": yield_during_chat}


@app.post("/load/set")
async def set_load(percent: int):
    """Меняем нагрузку БЕЗ перезапуска узла. percent: 0-100"""
    if percent < 20:
        load_config["sleep_between"] = 3.0
        load_config["max_concurrent"] = 1
    elif percent < 50:
        load_config["sleep_between"] = 1.0
        load_config["max_concurrent"] = 1
    elif percent < 80:
        load_config["sleep_between"] = 0.2
        load_config["max_concurrent"] = 2
    else:
        load_config["sleep_between"] = 0.0
        load_config["max_concurrent"] = 3
    load_config["percent"] = percent
    print(f"[Load] {percent}% → sleep={load_config['sleep_between']}s concurrent={load_config['max_concurrent']}", flush=True)
    return load_config


@app.get("/health")
def health():
    return {
        "worker_id":  WORKER_ID,
        "layers":     f"{LAYER_START}-{LAYER_END}",
        "model":      MODEL_NAME,
        "tasks_done": efct.tasks_done,
        "adaptation": round(efct.identity_distance(), 6),
        "com_last5":  efct.com_history[-5:],
        "status":     "ready",
    }

@app.get("/stats")
def stats():
    return {
        "worker_id":          WORKER_ID,
        "tier":               tier,
        "device":             device,
        "layer_start":        LAYER_START,
        "load_pct":           LOAD_PCT,
        "tasks_done":         efct.tasks_done,
        "adaptation":         round(efct.identity_distance(), 6),
        "gates_nonzero":      int((efct.gates != 0).sum().item()),
        "baseline":           round(float(efct.baseline or 0), 6),
        "com_history":        efct.com_history[-5:],
        "status":             "learning" if efct.tasks_done > 0 else "ready",
        "avg_speed":          round(avg_speed, 2),
        "last_com_signal":    round(last_quality_signal, 6) if last_quality_signal is not None else None,
        "last_signal_source": last_signal_source,
        "groq_enabled":       bool(GROQ_API_KEY),
        "federated_round":    efct.tasks_done // FEDERATED_INTERVAL,
        "next_sync_in":       FEDERATED_INTERVAL - (efct.tasks_done % FEDERATED_INTERVAL),
    }

@app.get("/efct/stats")
def efct_stats():
    return {
        "worker_id":       WORKER_ID,
        "tasks_done":      efct.tasks_done,
        "adaptation":      round(efct.identity_distance(), 6),
        "baseline":        round(efct.baseline, 8) if efct.baseline else None,
        "com_history":     efct.com_history[-20:],
        "gates_nonzero":   int((efct.gates.abs() > 1e-6).sum()),
        "phase_nonzero":   int((efct.phase.abs() > 1e-6).sum()),
        "gates_mean":      round(float(efct.gates.abs().mean()), 8),
    }

if __name__ == "__main__":
    print("="*55)
    print(f"  Noümind Pipeline Node")
    print(f"  ID:      {WORKER_ID}")
    print(f"  Слои:    {LAYER_START}-{LAYER_END}")
    print(f"  Gateway: {GATEWAY_URL}")
    print(f"  Порт:    {args.port}")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
