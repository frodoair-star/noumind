"""
gateway_pipeline.py — VPS
AirLLM backend: Mistral-7B-Instruct-v0.2 (layer streaming, ~0.4 GB RAM).
Раздаёт активации узлам через pull модель (distributed pipeline остаётся).
"""
import sys, os, io, asyncio, uuid, base64, time, re, json
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict

import torch
from transformers import AutoTokenizer
from airllm import AutoModel
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import websockets

try:
    import feedparser
    import httpx as _httpx_crawler
    from bs4 import BeautifulSoup
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    _CRAWLER_DEPS = True
except ImportError:
    _CRAWLER_DEPS = False
    print("[Crawler] Зависимости не установлены: pip install apscheduler feedparser httpx beautifulsoup4")

# ─────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────

MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"
LAYER_START  = 0
LAYER_END    = 31   # Mistral-7B: 32 decoder layers (0-31)
PORT         = 8002
WS_PORT      = 8003  # Reverse WebSocket — узлы подключаются сюда сами

# ─────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ — AirLLM (layer streaming)
# ─────────────────────────────────────────

print(f"\n[Gateway] Загружаю {MODEL_NAME} через AirLLM (CPU, fp16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(
    MODEL_NAME,
    device="cpu",
    dtype=torch.float16,
    max_seq_len=512,
)

total_layers = 32  # Mistral-7B: 32 decoder layers

print(f"[Gateway] AirLLM готов | {MODEL_NAME}")
print(f"[Gateway] Decoder layers: {total_layers}")
print(f"[Gateway] RAM: ~0.4 GB (веса стримятся с диска)")
print(f"[Gateway] Готов!\n")

# ─────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()

def bytes_to_tensor(data: bytes) -> torch.Tensor:
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=False)

def forward_first_half(input_ids: torch.Tensor) -> torch.Tensor:
    """Stub: в AirLLM-режиме gateway делает полный inference сам.
    Возвращает нули нужной формы, чтобы не ломать crawler."""
    batch, seq = input_ids.shape
    return torch.zeros(batch, seq, 4096)   # Mistral hidden_size = 4096

# ─────────────────────────────────────────
# ГЕНЕРАЦИЯ ОТВЕТА — AirLLM (greedy decoding)
# ─────────────────────────────────────────

def generate_response(message: str, max_new_tokens: int = 300) -> str:
    """
    Полный inference через AirLLM (greedy, no cache).
    Каждый шаг стримит все 35 layer-файлов с диска → ~12-15с/токен на CPU.
    """
    prompt = f"[INST] {SYSTEM_PROMPT}\n\n{message} [/INST]"
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=512
    )

    generated   = input_ids.clone()
    eos_id      = tokenizer.eos_token_id or 2
    prompt_len  = input_ids.shape[1]

    print(f"[AirLLM] Промпт: {prompt_len} токенов | цель: {max_new_tokens} новых")

    with torch.no_grad():
        for step in range(max_new_tokens):
            t0  = time.time()
            out = model(generated)

            # AirLLM возвращает CausalLMOutputWithCrossAttentions или tuple
            logits = out.logits if hasattr(out, "logits") else out[0]
            # logits: [1, seq_len, vocab_size]

            next_token_id = int(logits[:, -1, :].argmax(-1).item())
            token_text    = tokenizer.decode([next_token_id])
            elapsed       = time.time() - t0

            print(
                f"[AirLLM] [{step+1:2d}/{max_new_tokens}] "
                f"tok={next_token_id:6d} '{token_text}' ({elapsed:.1f}s)",
                flush=True
            )

            if next_token_id == eos_id:
                print(f"[AirLLM] EOS на шаге {step+1}")
                break

            generated = torch.cat(
                [generated, torch.tensor([[next_token_id]])], dim=1
            )

    new_ids = generated[0][prompt_len:].tolist()
    result  = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    print(f"[AirLLM] Готово: '{result}'")
    return result

# ─────────────────────────────────────────
# ОЧЕРЕДИ + WORKER REGISTRY — pull модель
# ─────────────────────────────────────────

pending_activations: deque = deque()
result_store: Dict[str, str] = {}
worker_stats: Dict[str, dict] = {}
streaming_queues: Dict[str, asyncio.Queue] = {}

# ─────────────────────────────────────────
# Reverse WebSocket — узлы подключаются к Gateway сами
# ─────────────────────────────────────────
reverse_ws_connections: Dict[str, any] = {}  # worker_id → websocket
result_events: Dict[str, asyncio.Event] = {} # task_id  → Event (сигнал готовности)


async def handle_node_connection(websocket):
    """
    Узел сам подключается сюда (reverse WS).
    Первое сообщение — {"type":"register", "worker_id":...}.
    Gateway держит соединение открытым и шлёт задачи.
    """
    worker_id = None
    try:
        # ── 1. Регистрация ───────────────────────────────
        raw  = await asyncio.wait_for(websocket.recv(), timeout=30)
        data = json.loads(raw)

        if data.get("type") != "register":
            await websocket.close(1008, "First message must be register")
            return

        worker_id = data["worker_id"]
        reverse_ws_connections[worker_id] = websocket

        # Обновляем/создаём статус узла
        prev = worker_stats.get(worker_id, {})
        worker_stats[worker_id] = {
            **prev,
            "status":    "ready",
            "tier":      data.get("tier",    "slow_cpu"),
            "device":    data.get("device",  "cpu"),
            "ram_gb":    data.get("ram_gb",  4.0),
            "vram_gb":   data.get("vram_gb", 0.0),
            "model":     data.get("model",   ""),
            "backend":   "airllm_ws",
            "last_seen": time.time(),
        }

        # Назначаем слои если ещё не назначены
        if worker_id not in layer_assignments:
            effective = data.get("vram_gb") or data.get("ram_gb", 4.0)
            assign_layers(worker_id, effective)

        ls, le = layer_assignments.get(worker_id, (0, 31))
        worker_stats[worker_id]["layer_start"] = ls
        worker_stats[worker_id]["layer_end"]   = le

        print(f"[Gateway] ✓ Узел подключён (reverse WS): {worker_id} "
              f"({data.get('tier')}) слои {ls}-{le} "
              f"покрытие {get_coverage()['percent']}%")

        # ── 2. Подтверждаем регистрацию ──────────────────
        await websocket.send(json.dumps({
            "type":        "registered",
            "worker_id":   worker_id,
            "layer_start": ls,
            "layer_end":   le,
            "model":       MODEL_NAME,
            "coverage":    get_coverage(),
        }))

        # ── 3. Цикл приёма сообщений от узла ─────────────
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "result":
                task_id = data.get("task_id", "")
                text    = data.get("text", "")
                result_store[task_id] = text
                ev = result_events.get(task_id)
                if ev:
                    ev.set()
                print(f"[Gateway] ← результат {task_id[:8]} "
                      f"от {worker_id}: '{text[:50]}'")

            elif msg_type == "heartbeat":
                worker_stats[worker_id]["last_seen"] = time.time()
                await websocket.send(json.dumps({"type": "pong"}))

            elif msg_type == "ready":
                # Узел сообщает что модель загружена
                worker_stats[worker_id]["status"] = "ready"
                print(f"[Gateway] {worker_id} → модель готова")

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[Gateway] WS ошибка ({worker_id}): {e}")
    finally:
        if worker_id:
            reverse_ws_connections.pop(worker_id, None)
            if worker_id in worker_stats:
                worker_stats[worker_id]["status"] = "offline"
            print(f"[Gateway] Узел отключился: {worker_id}")


def _ws_is_open(ws) -> bool:
    """Проверяем что соединение открыто (совместимо с websockets 10-16)."""
    # websockets >=13 убрал .closed; используем .open или state
    if hasattr(ws, 'open'):
        return ws.open
    if hasattr(ws, 'state'):
        from websockets.connection import State
        return ws.state == State.OPEN
    # Fallback: если объект в словаре — считаем живым
    return True


async def send_task_to_worker(worker_id: str, task: dict) -> str:
    """
    Шлём задачу узлу через обратное WS-соединение.
    Ждём ответ через asyncio.Event (не polling).
    """
    ws = reverse_ws_connections.get(worker_id)
    if ws is None:
        raise Exception(f"Узел {worker_id} не подключён")

    task_id = task["task_id"]
    ev = asyncio.Event()
    result_events[task_id] = ev
    result_store[task_id]  = None

    try:
        await ws.send(json.dumps(task))
        await asyncio.wait_for(ev.wait(), timeout=120.0)
        return result_store.pop(task_id, "")
    except asyncio.TimeoutError:
        raise Exception(f"Timeout: узел {worker_id} не ответил за 120с")
    finally:
        result_events.pop(task_id, None)
        result_store.pop(task_id, None)


async def start_ws_server():
    """Запускаем WebSocket-сервер для обратных подключений узлов."""
    print(f"[Gateway] Reverse WebSocket сервер на порту {WS_PORT}")
    async with websockets.serve(
        handle_node_connection,
        "0.0.0.0",
        WS_PORT,
        ping_interval = 20,
        ping_timeout  = 60,
    ):
        await asyncio.Future()  # держим до остановки event loop

# ─────────────────────────────────────────
# ФЕДЕРАТИВНОЕ ОБУЧЕНИЕ — gates store
# ─────────────────────────────────────────
gates_store: Dict[str, dict] = {}

# ─────────────────────────────────────────
# DISTRIBUTED PIPELINE — назначение слоёв
# ─────────────────────────────────────────

DISTRIBUTED_MODEL = {
    "id":           "mistralai/Mistral-7B-Instruct-v0.2",
    "total_layers": 32,
    "layers_per_gb": 2.5,   # слоёв на 1 GB RAM/VRAM
}

layer_assignments: Dict[str, tuple] = {}   # worker_id → (start, end)


def assign_layers(worker_id: str, ram_gb: float) -> tuple:
    """Назначаем узлу незанятые слои, пропорционально его RAM."""
    total = DISTRIBUTED_MODEL["total_layers"]

    occupied: set = set()
    for wid, (s, e) in layer_assignments.items():
        if wid != worker_id:
            for ll in range(s, e + 1):
                occupied.add(ll)

    my_count = max(1, min(
        int(ram_gb * DISTRIBUTED_MODEL["layers_per_gb"]),
        total - len(occupied),
    ))

    # Первый свободный непрерывный диапазон
    start = 0
    while start < total and start in occupied:
        start += 1

    end = min(start + my_count - 1, total - 1)

    # Если диапазон пересекается — сдвигаем дальше
    while any(ll in occupied for ll in range(start, end + 1)) and start < total:
        start = end + 1
        end   = min(start + my_count - 1, total - 1)
    if start >= total:
        start = 0
        end   = min(my_count - 1, total - 1)

    layer_assignments[worker_id] = (start, end)
    print(f"[Gateway] {worker_id[:16]}: слои {start}-{end} "
          f"({my_count} сл, {ram_gb:.1f} GB)")
    return start, end


def build_pipeline_chain() -> list:
    """Цепочка узлов, отсортированная по layer_start."""
    chain = []
    for wid, (s, e) in sorted(layer_assignments.items(), key=lambda x: x[1][0]):
        if wid in worker_stats and worker_stats[wid].get("status") in ("free", "ready"):
            chain.append({
                "worker_id":   wid,
                "layer_start": s,
                "layer_end":   e,
                "tier":        worker_stats[wid].get("tier", "slow_cpu"),
            })
    return chain


def get_coverage() -> dict:
    """Какой процент слоёв модели покрыт."""
    total   = DISTRIBUTED_MODEL["total_layers"]
    covered = set()
    for s, e in layer_assignments.values():
        for ll in range(s, e + 1):
            covered.add(ll)
    return {
        "covered": len(covered),
        "total":   total,
        "percent": round(len(covered) / total * 100),
        "missing": [ll for ll in range(total) if ll not in covered],
    }

# ─────────────────────────────────────────
# НОЧНОЙ КРАУЛЕР — источники и состояние
# ─────────────────────────────────────────

SOURCES = {
    "wikipedia": "https://en.wikipedia.org/api/rest_v1/page/random/summary",
    "arxiv":     "http://export.arxiv.org/rss/cs.AI",
    "hackernews": "https://hnrss.org/frontpage",
    "bbc":       "http://feeds.bbci.co.uk/news/rss.xml",
    "techcrunch": "https://techcrunch.com/feed/",
}

training_load: Dict[str, int] = {}

crawler_stats = {
    "chunks_processed": 0,
    "sources_visited":  0,
    "last_run":         None,
    "running":          False,
    "errors":           0,
}


async def fetch_wikipedia() -> str:
    if not _CRAWLER_DEPS: return ""
    try:
        async with _httpx_crawler.AsyncClient(timeout=10) as c:
            r = await c.get(SOURCES["wikipedia"])
            d = r.json()
            return f"{d.get('title','')}: {d.get('extract','')}"
    except Exception as e:
        print(f"[Crawler] wikipedia error: {e}")
        return ""

async def fetch_rss(name: str, url: str) -> list:
    if not _CRAWLER_DEPS: return []
    try:
        async with _httpx_crawler.AsyncClient(timeout=10) as c:
            r = await c.get(url)
        feed = feedparser.parse(r.text)
        texts = []
        for entry in feed.entries[:5]:
            raw = entry.get("summary", "") or entry.get("title", "")
            text = BeautifulSoup(raw, "html.parser").get_text().strip()
            if len(text) > 100:
                texts.append(text)
        return texts
    except Exception as e:
        print(f"[Crawler] {name} rss error: {e}")
        return []

async def fetch_webpage(url: str) -> str:
    if not _CRAWLER_DEPS: return ""
    try:
        async with _httpx_crawler.AsyncClient(timeout=15, follow_redirects=True) as c:
            r = await c.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text)[:3000]
    except Exception as e:
        print(f"[Crawler] fetch_webpage error: {e}")
        return ""

def split_into_chunks(text: str, chunk_size: int = 150) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def get_least_loaded_worker() -> str | None:
    active = [wid for wid, s in worker_stats.items() if s.get("status") == "free"]
    if not active:
        return None
    return min(active, key=lambda wid: training_load.get(wid, 0))

async def send_training_chunk(chunk: str, source: str) -> bool:
    try:
        chat_msgs = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": chunk},
        ]
        formatted = tokenizer.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"]

        import concurrent.futures
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            activations = await loop.run_in_executor(pool, forward_first_half, input_ids)
        act_bytes = tensor_to_bytes(activations)

        task_id = str(uuid.uuid4())
        pending_activations.append({
            "task_id":     task_id,
            "act_bytes":   act_bytes,
            "input_shape": str(tuple(activations.shape)),
            "seq_len":     activations.shape[1],
            "message":     chunk[:200],
            "input_ids":   input_ids.tolist()[0],
            "priority":    "economy",
            "training":    True,
            "source":      source,
        })

        wid = get_least_loaded_worker()
        if wid:
            training_load[wid] = training_load.get(wid, 0) + 1
            print(f"[Crawler] → {wid} | {source} | нагрузка: {training_load[wid]}", flush=True)
        return True

    except Exception as e:
        print(f"[Crawler] send_training_chunk error: {e}", flush=True)
        crawler_stats["errors"] += 1
        return False


async def crawl_and_train():
    if crawler_stats["running"]:
        print("[Crawler] Уже запущен, пропускаю")
        return
    if not worker_stats:
        print("[Crawler] Нет зарегистрированных узлов, пропускаю")
        return

    crawler_stats["running"]  = True
    crawler_stats["last_run"] = time.time()
    print("[Crawler] ═══ Начинаю сбор данных ═══", flush=True)

    try:
        all_chunks = []

        for _ in range(3):
            text = await fetch_wikipedia()
            if text:
                all_chunks += [("wikipedia", c) for c in split_into_chunks(text)]

        for name, url in SOURCES.items():
            if name == "wikipedia":
                continue
            texts = await fetch_rss(name, url)
            for t in texts:
                all_chunks += [(name, c) for c in split_into_chunks(t)]

        print(f"[Crawler] Собрано {len(all_chunks)} чанков из {len(SOURCES)} источников", flush=True)

        sent = 0
        for source, chunk in all_chunks:
            ok = await send_training_chunk(chunk, source)
            if ok:
                sent += 1
                crawler_stats["chunks_processed"] += 1
            await asyncio.sleep(0.5)

        crawler_stats["sources_visited"] += len(SOURCES)
        print(f"[Crawler] ═══ Готово. Отправлено: {sent}/{len(all_chunks)} чанков ═══", flush=True)

    except Exception as e:
        print(f"[Crawler] Ошибка: {e}", flush=True)
        crawler_stats["errors"] += 1
    finally:
        crawler_stats["running"] = False


COMPUTE_COST = {
    "gpu":      10,
    "fast_cpu": 4,
    "slow_cpu": 1,
}

def get_best_worker(priority: str = "balanced") -> dict | None:
    free = [w for w in worker_stats.values() if w.get("status", "free") == "free"]
    if not free:
        return None
    if priority == "fast":
        order = {"gpu": 0, "fast_cpu": 1, "slow_cpu": 2}
        return min(free, key=lambda w: (order.get(w["tier"], 3), w.get("avg_speed", 999)))
    elif priority == "economy":
        order = {"slow_cpu": 0, "fast_cpu": 1, "gpu": 2}
        return min(free, key=lambda w: order.get(w["tier"], 3))
    else:
        return min(free, key=lambda w: w.get("avg_speed", 999))

# ─────────────────────────────────────────
# MoE ROUTER (по образцу DeepSeek)
# ─────────────────────────────────────────

EXPERT_CATEGORIES = {
    "code":     ["code", "python", "function", "bug",
                 "error", "programming", "script", "debug",
                 "javascript", "api", "database", "sql"],
    "science":  ["physics", "chemistry", "biology", "math",
                 "equation", "formula", "research", "theory",
                 "quantum", "molecule", "experiment"],
    "history":  ["history", "war", "ancient", "century",
                 "historical", "civilization", "empire"],
    "creative": ["story", "poem", "write", "creative",
                 "imagine", "fiction", "novel", "song"],
    "general":  [],
}

def classify_request(message: str) -> list:
    msg = message.lower()
    matched = [cat for cat, kws in EXPERT_CATEGORIES.items()
               if cat != "general" and any(kw in msg for kw in kws)]
    return matched or ["general"]

def get_expert_workers(categories: list) -> list:
    experts = []
    for worker_id, stats in worker_stats.items():
        if stats.get("status", "free") != "free":
            continue
        if stats.get("is_fallback", False):
            experts.append({"id": worker_id, "type": "shared", "score": 0.5})
            continue
        spec = stats.get("specialization", "general")
        if spec in categories:
            score = stats.get("category_scores", {}).get(spec, 0)
            experts.append({"id": worker_id, "type": "routed", "score": score})
    experts.sort(key=lambda x: x["score"], reverse=True)
    return experts

def select_moe_worker(message: str, priority: str) -> tuple:
    categories = classify_request(message)
    experts    = get_expert_workers(categories)
    info = {"categories": categories, "decision": "—", "worker_id": None, "score": None}

    selected = None
    if experts:
        if priority == "fast":
            selected = min(experts, key=lambda e: worker_stats[e["id"]].get("avg_speed", 999))
            info["decision"] = f"fastest expert ({selected['type']})"
        elif priority == "economy":
            shared = [e for e in experts if e["type"] == "shared"]
            selected = shared[0] if shared else experts[0]
            info["decision"] = "shared (economy)" if shared else "cheapest expert"
        else:
            routed = [e for e in experts if e["type"] == "routed"]
            if routed:
                selected = routed[0]
                info["decision"] = "routed expert"
            else:
                shared = [e for e in experts if e["type"] == "shared"]
                selected = shared[0] if shared else None
                info["decision"] = "shared (fallback)"

    if selected is None:
        any_free = get_best_worker(priority)
        if any_free:
            for wid, s in worker_stats.items():
                if s is any_free:
                    selected = {"id": wid, "type": "any", "score": 0.0}
                    info["decision"] = "any free node (no expert)"
                    break

    if selected:
        info["worker_id"] = selected["id"]
        info["score"]     = selected.get("score")
    return (selected["id"] if selected else None), info

FALLBACK_SEC = 8.0

def _pop_task_for_worker(worker_id: str):
    if not pending_activations:
        return None
    now = time.time()
    for i, t in enumerate(pending_activations):
        if t.get("target_worker") == worker_id:
            del pending_activations[i]; return t
    for i, t in enumerate(pending_activations):
        if t.get("target_worker") is None:
            del pending_activations[i]; return t
    for i, t in enumerate(pending_activations):
        if now - t.get("enqueued_at", now) > FALLBACK_SEC:
            print(f"[Router] ⚠ задача {t['task_id'][:8]} просрочена → отдаю {worker_id}", flush=True)
            del pending_activations[i]; return t
    return None

# ─────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────

scheduler = AsyncIOScheduler() if _CRAWLER_DEPS else None

@asynccontextmanager
async def lifespan(application: FastAPI):
    # Запускаем Reverse WebSocket сервер для узлов
    asyncio.create_task(start_ws_server())
    print(f"[Gateway] Reverse WS сервер запущен (порт {WS_PORT})")

    if scheduler and _CRAWLER_DEPS:
        scheduler.add_job(crawl_and_train, "cron", hour=2, minute=0, id="nightly_crawl")
        scheduler.add_job(
            crawl_and_train, "date",
            run_date=datetime.now() + timedelta(seconds=30),
            id="startup_crawl",
        )
        scheduler.start()
        print("[Crawler] Планировщик запущен (ночной краулер в 02:00 + старт через 30 сек)")
    yield
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)

SYSTEM_PROMPT = """You are a helpful multilingual AI assistant.
IMPORTANT: Always respond in the same language as the user's question.
If the user writes in Russian, respond in Russian.
If the user writes in English, respond in English.
Be concise and helpful. Do not repeat the question."""

app = FastAPI(title="Noümind Pipeline Gateway", lifespan=lifespan)

@app.post("/chat")
async def chat(message: str, session_id: str = "default", priority: str = "balanced"):
    """
    Полный inference через AirLLM (Mistral-7B, greedy, layer streaming).
    ⚠️  Медленно на CPU: ~12-15с/токен × 20 токенов ≈ 4-5 минут.
    Используй curl --max-time 600.
    """
    task_id = str(uuid.uuid4())
    print(f"\n[Gateway] ← '{message[:60]}'")

    import concurrent.futures
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        response_text = await loop.run_in_executor(pool, generate_response, message)

    print(f"[Gateway] ✓ '{response_text[:80]}'\n")
    return {
        "response":   response_text,
        "task_id":    task_id,
        "session_id": session_id,
        "model":      MODEL_NAME,
        "backend":    "airllm",
        "layers":     f"{LAYER_START}-{LAYER_END}",
    }


@app.get("/activations/next")
async def get_next_activations(worker_id: str = "unknown"):
    """Узел забирает задачу (distributed pipeline — для будущих узлов)."""
    task = _pop_task_for_worker(worker_id)
    if task is None:
        return {"task_id": None}

    task_id     = task["task_id"]
    act_bytes   = task["act_bytes"]
    input_shape = task["input_shape"]

    tgt  = task.get("target_worker")
    mark = "✓ свой" if tgt == worker_id else ("⚡ fallback" if tgt else "общая")
    print(f"[Gateway] → Задача {task_id[:8]} → {worker_id} [{mark}] | shape={input_shape}")

    return {
        "task_id":     task_id,
        "activations": base64.b64encode(act_bytes).decode(),
        "shape":       input_shape,
        "layer_start": LAYER_END + 1,
        "message":     task.get("message", ""),
        "input_ids":   task.get("input_ids", []),
        "streaming":   task.get("streaming", False),
    }


@app.post("/activations/result")
async def submit_result(
    task_id:    str,
    result:     str,
    confidence: float = 0.0,
    worker_id:  str   = "unknown",
    avg_speed:  float = 0.0,
):
    result_store[task_id] = {"text": result, "worker_id": worker_id}
    if task_id in streaming_queues:
        await streaming_queues[task_id].put(None)
    if worker_id in worker_stats:
        worker_stats[worker_id]["tasks_done"] = worker_stats[worker_id].get("tasks_done", 0) + 1
        worker_stats[worker_id]["status"]     = "free"
        if avg_speed > 0:
            worker_stats[worker_id]["avg_speed"] = avg_speed
    print(f"[Gateway] ✓ Результат от {worker_id}: '{result[:60]}' | conf={confidence:.6f}")
    return {"status": "ok"}


class RegisterPayload(BaseModel):
    category_scores: dict | None = None


@app.post("/register")
async def register_worker(
    request:        Request,
    worker_id:      str,
    tier:           str   = "slow_cpu",
    device:         str   = "cpu",
    ram_gb:         float = 0,
    vram_gb:        float = 0,
    layer_start:    int   = -1,
    layer_end:      int   = -1,
    model:          str   = "",
    backend:        str   = "",
    specialization: str   = "general",
    is_fallback:    bool  = False,
    ws_port:        int   = 9010,
    payload:        RegisterPayload | None = None,
):
    prev      = worker_stats.get(worker_id, {})
    client_ip = request.client.host if request.client else "unknown"

    worker_stats[worker_id] = {
        "tier":            tier,
        "device":          device,
        "ram_gb":          ram_gb,
        "vram_gb":         vram_gb,
        "status":          "ready",
        "tasks_done":      prev.get("tasks_done", 0),
        "avg_speed":       prev.get("avg_speed", 999),
        "specialization":  specialization,
        "is_fallback":     is_fallback,
        "backend":         backend,
        "category_scores": (payload.category_scores if payload and payload.category_scores
                            else prev.get("category_scores", {})),
        # WebSocket — для Gateway → Node соединений
        "ws_port":         ws_port,
        "host":            client_ip,
        "last_seen":       time.time(),
    }

    # Sync layer assignment if node tells us its layers
    if layer_start >= 0 and layer_end >= 0:
        layer_assignments[worker_id] = (layer_start, layer_end)

    coverage = get_coverage()
    role = "Shared Expert" if is_fallback else f"Routed Expert [{specialization}]"
    print(f"[Gateway] ✓ {worker_id[:20]} | {role} | tier={tier} | "
          f"слои {layer_start}-{layer_end} | ws={client_ip}:{ws_port} | "
          f"покрытие {coverage['percent']}%")
    return {
        "status":         "registered",
        "worker_id":       worker_id,
        "specialization":  specialization,
        "coverage":        coverage,
    }


# ── Pipeline pull-endpoints (для distributed узлов) ───────────────────────────

@app.get("/pipeline/task")
async def pipeline_get_task(worker_id: str = "unknown", layer_start: int = 0, layer_end: int = 31):
    task = _pop_task_for_worker(worker_id)
    if task is None:
        return {"task_id": None}

    task_id     = task["task_id"]
    act_bytes   = task["act_bytes"]
    input_shape = task["input_shape"]
    print(f"[Gateway] pipeline/task → {worker_id} | {task_id[:8]}")

    return {
        "task_id":     task_id,
        "activations": base64.b64encode(act_bytes).decode(),
        "shape":       input_shape,
        "layer_start": layer_start,
        "message":     task.get("message", ""),
        "input_ids":   task.get("input_ids", []),
    }


@app.get("/pipeline/activations/{task_id}")
async def pipeline_get_activations(task_id: str):
    for t in pending_activations:
        if t["task_id"] == task_id:
            return t["act_bytes"]
    return b""


@app.post("/pipeline/submit")
async def pipeline_submit(task_id: str, worker_id: str = "unknown", layer_end: int = 31,
                          body: bytes = b""):
    # Сохраняем bytes — нода возвращает логиты
    result_store[task_id] = {"bytes": body, "worker_id": worker_id}
    if worker_id in worker_stats:
        worker_stats[worker_id]["status"] = "free"
    print(f"[Gateway] pipeline/submit ← {worker_id} | {task_id[:8]} | {len(body):,} bytes")
    return {"status": "ok"}


@app.post("/activations/token")
async def push_token(task_id: str, token: str):
    q = streaming_queues.get(task_id)
    if q:
        await q.put(token)
    return {"ok": True}


@app.get("/chat/stream")
async def chat_stream(message: str, session_id: str = "default", priority: str = "balanced"):
    """
    SSE true-streaming: каждый токен отправляется клиенту сразу как готов.
    Первый токен — ~30с, дальше потоком. Требует таймаут клиента ≥ 10 мин.
    """
    task_id = str(uuid.uuid4())
    print(f"\n[Gateway] ← STREAM '{message[:60]}'")

    loop      = asyncio.get_running_loop()
    token_q: asyncio.Queue = asyncio.Queue()

    def _generate_streaming(msg: str, max_new_tokens: int = 300):
        """Запускается в потоке. Каждый сгенерированный токен кладёт в token_q."""
        prompt    = f"[INST] {SYSTEM_PROMPT}\n\n{msg} [/INST]"
        input_ids = tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        generated = input_ids.clone()
        eos_id    = tokenizer.eos_token_id or 2

        print(f"[AirLLM] STREAM промпт: {input_ids.shape[1]} токенов")

        try:
            with torch.no_grad():
                for step in range(max_new_tokens):
                    t0  = time.time()
                    out = model(generated)
                    logits = out.logits if hasattr(out, "logits") else out[0]
                    next_id   = int(logits[:, -1, :].argmax(-1).item())
                    tok_text  = tokenizer.decode([next_id], skip_special_tokens=False)
                    # strip special tokens manually
                    tok_clean = tokenizer.decode([next_id], skip_special_tokens=True)
                    elapsed   = time.time() - t0

                    print(
                        f"[AirLLM] STREAM [{step+1:2d}/{max_new_tokens}] "
                        f"tok={next_id} '{tok_clean}' ({elapsed:.1f}s)",
                        flush=True
                    )

                    # Отправляем токен в очередь (из потока → event loop)
                    asyncio.run_coroutine_threadsafe(
                        token_q.put(tok_clean), loop
                    )

                    if next_id == eos_id:
                        print(f"[AirLLM] STREAM EOS на шаге {step+1}")
                        break

                    generated = torch.cat(
                        [generated, torch.tensor([[next_id]])], dim=1
                    )
        except Exception as e:
            print(f"[AirLLM] STREAM ошибка: {e}", flush=True)
        finally:
            # Сигнал конца
            asyncio.run_coroutine_threadsafe(token_q.put(None), loop)

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(_generate_streaming, message)

    async def event_gen():
        try:
            while True:
                try:
                    token = await asyncio.wait_for(token_q.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    print(f"[AirLLM] STREAM токен не пришёл за 120с → [DONE]")
                    yield "data: [DONE]\n\n"
                    return

                if token is None:
                    yield "data: [DONE]\n\n"
                    return

                if token:   # пропускаем пустые токены
                    safe = token.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"
        finally:
            executor.shutdown(wait=False)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class GatesPayload(BaseModel):
    gates: list
    phase: list


@app.post("/gates/upload")
async def upload_gates(worker_id: str, tasks_done: int, payload: GatesPayload):
    gates_store[worker_id] = {
        "gates":      torch.tensor(payload.gates, dtype=torch.float32),
        "phase":      torch.tensor(payload.phase, dtype=torch.float32),
        "tasks":      tasks_done,
        "updated_at": time.time(),
    }

    contributors = {k: v for k, v in gates_store.items() if k != "__global__"}
    if len(contributors) >= 2:
        all_gates   = [v["gates"] for v in contributors.values()]
        all_phase   = [v["phase"] for v in contributors.values()]
        task_counts = torch.tensor([v["tasks"] for v in contributors.values()], dtype=torch.float32)
        weights     = task_counts / task_counts.sum()

        global_gates = sum(w * g for w, g in zip(weights, all_gates))
        global_phase = sum(w * p for w, p in zip(weights, all_phase))

        gates_store["__global__"] = {
            "gates":        global_gates,
            "phase":        global_phase,
            "contributors": len(contributors),
            "updated_at":   time.time(),
        }
        print(f"[Federated] ✓ Глобальные gates обновлены | участников: {len(contributors)}")

    return {"status": "ok", "contributors": len(contributors)}


@app.get("/gates/global")
async def get_global_gates():
    if "__global__" not in gates_store:
        return {"status": "not_ready", "contributors": 0}
    g = gates_store["__global__"]
    return {
        "status":       "ok",
        "gates":        g["gates"].tolist(),
        "phase":        g["phase"].tolist(),
        "contributors": g["contributors"],
        "updated_at":   g["updated_at"],
    }


@app.get("/gates/stats")
async def gates_stats():
    contributors = {
        k: {"tasks": v["tasks"], "updated_at": round(v["updated_at"], 1)}
        for k, v in gates_store.items()
        if k != "__global__"
    }
    global_info = None
    if "__global__" in gates_store:
        g = gates_store["__global__"]
        global_info = {
            "contributors": g["contributors"],
            "updated_at":   round(g["updated_at"], 1),
        }
    return {
        "total_contributors": len(contributors),
        "global_ready":       "__global__" in gates_store,
        "global":             global_info,
        "contributors":       contributors,
    }


@app.post("/crawler/run")
async def run_crawler():
    if crawler_stats["running"]:
        return {"status": "already_running", "chunks_processed": crawler_stats["chunks_processed"]}
    asyncio.create_task(crawl_and_train())
    return {"status": "started"}


@app.get("/crawler/stats")
async def get_crawler_stats():
    return {
        **crawler_stats,
        "last_run_ago":      round(time.time() - crawler_stats["last_run"], 1) if crawler_stats["last_run"] else None,
        "training_load":     training_load,
        "queue_size":        len(pending_activations),
        "workers_available": len([w for w, s in worker_stats.items() if s.get("status") == "free"]),
    }


@app.post("/crawler/fetch")
async def fetch_url_endpoint(url: str):
    text = await fetch_webpage(url)
    if not text:
        return {"status": "error", "message": "Не удалось загрузить страницу"}
    chunks = split_into_chunks(text)
    sent = 0
    for chunk in chunks:
        ok = await send_training_chunk(chunk, url)
        if ok:
            sent += 1
            crawler_stats["chunks_processed"] += 1
        await asyncio.sleep(0.1)
    return {"status": "ok", "url": url, "chunks_total": len(chunks), "chunks_sent": sent}


def _current_model(n: int) -> str:
    if n >= 50: return "Llama-3-70B"
    if n >= 10: return "Llama-3-8B"
    return "Mistral-7B-Instruct"

def _next_model(n: int) -> str:
    if n >= 50: return "—"
    if n >= 10: return "Llama-3-70B"
    return "Llama-3-8B"

def _nodes_needed(n: int) -> int:
    if n >= 50: return 0
    if n >= 10: return 50 - n
    return 10 - n


@app.get("/network")
async def network():
    n = len(worker_stats)
    active = [wid for wid, s in worker_stats.items() if s.get("status") in ("free", "ready", None) or "tier" in s]

    expert_map = {}
    for wid, s in worker_stats.items():
        spec = s.get("specialization", "general")
        expert_map.setdefault(spec, []).append({
            "id":          wid[:12],
            "score":       round(s.get("category_scores", {}).get(spec, 0), 4),
            "is_fallback": s.get("is_fallback", False),
            "tier":        s.get("tier", "slow_cpu"),
        })

    return {
        "total_workers":            n,
        "active_workers":           len(active),
        "current_model":            _current_model(n),
        "next_model":               _next_model(n),
        "nodes_needed_for_upgrade": _nodes_needed(n),
        "expert_map":               expert_map,
        "router_active":            True,
        "categories":               list(EXPERT_CATEGORIES.keys()),
        "queue_size":               len(pending_activations),
        "federated_contributors":   len([k for k in gates_store if k != "__global__"]),
        "backend":                  "airllm",
        "inference_model":          MODEL_NAME,
    }


# ─────────────────────────────────────────
# DISTRIBUTED PIPELINE — endpoints
# ─────────────────────────────────────────

@app.post("/assign_layers")
async def assign_layers_endpoint(
    worker_id: str,
    ram_gb:    float,
    vram_gb:   float = 0.0,
):
    """Узел регистрируется и получает назначение слоёв."""
    effective = vram_gb if vram_gb > 0 else ram_gb
    start, end = assign_layers(worker_id, effective)
    coverage   = get_coverage()
    return {
        "layer_start": start,
        "layer_end":   end,
        "layer_count": end - start + 1,
        "model_id":    DISTRIBUTED_MODEL["id"],
        "coverage":    coverage,
        "message":     f"Загрузи слои {start}-{end} через AirLLM",
    }


@app.post("/pipeline/chat")
async def pipeline_chat(message: str, session_id: str = "default"):
    """
    Запрос идёт через цепочку узлов. Если покрытие < 80% — падаем на VPS AirLLM.
    """
    coverage = get_coverage()
    chain    = build_pipeline_chain()

    # Недостаточно узлов — прямой VPS inference
    if coverage["percent"] < 80 or not chain:
        print(f"[Pipeline] Покрытие {coverage['percent']}% — VPS fallback")
        return await chat(message, session_id=session_id)

    # Кладём задачу в очередь для первого узла цепочки
    task_id       = str(uuid.uuid4())
    target_worker = chain[0]["worker_id"]
    print(f"[Pipeline] '{message[:40]}' → {target_worker[:16]} "
          f"(покрытие {coverage['percent']}%)")

    pending_activations.append({
        "task_id":       task_id,
        "act_bytes":     b"",          # узел получает сообщение, не активации
        "input_shape":   "(0,)",
        "seq_len":       0,
        "message":       message,
        "input_ids":     [],
        "priority":      "balanced",
        "target_worker": target_worker,
        "enqueued_at":   time.time(),
    })

    # Ждём результат от узла (до 600 с для AirLLM)
    for _ in range(1200):
        if task_id in result_store:
            entry = result_store.pop(task_id)
            text  = entry["text"] if isinstance(entry, dict) else entry
            wid   = entry.get("worker_id", target_worker) if isinstance(entry, dict) else target_worker
            print(f"[Pipeline] ✓ от {wid[:16]}: '{text[:60]}'")
            return {
                "response":  text,
                "task_id":   task_id,
                "session_id": session_id,
                "worker_id": wid,
                "coverage":  coverage,
                "pipeline":  True,
            }
        await asyncio.sleep(0.5)

    # Таймаут — убираем из очереди
    for i, t in enumerate(list(pending_activations)):
        if t.get("task_id") == task_id:
            try: del pending_activations[i]
            except: pass
            break
    return {"error": "pipeline timeout — no node responded"}


@app.get("/pipeline/status")
async def pipeline_status():
    """Статус распределённого pipeline."""
    chain    = build_pipeline_chain()
    coverage = get_coverage()
    return {
        "chain":             chain,
        "coverage":          coverage,
        "model":             DISTRIBUTED_MODEL["id"],
        "ready":             coverage["percent"] >= 80,
        "assignments":       {wid: {"start": s, "end": e}
                              for wid, (s, e) in layer_assignments.items()},
        "registered_workers": len(worker_stats),
    }


@app.post("/heartbeat")
async def heartbeat(worker_id: str):
    """Узел сообщает что жив."""
    if worker_id in worker_stats:
        worker_stats[worker_id]["last_seen"] = time.time()
        worker_stats[worker_id]["status"]    = "ready"
    return {"ok": True}


@app.post("/ws/chat")
async def ws_chat(message: str, priority: str = "balanced"):
    """
    Запрос через Reverse WebSocket (узел сам подключён к Gateway).
    Выбирает лучший подключённый узел по tier.
    Если узлов нет — VPS AirLLM fallback.
    """
    # Выбираем подключённые reverse-WS узлы
    # (если воркер в словаре — соединение живо; handle_node_connection
    #  удаляет его из reverse_ws_connections при отключении)
    available = [
        wid for wid in reverse_ws_connections
        if worker_stats.get(wid, {}).get("status") == "ready"
    ]

    if not available:
        print(f"[Gateway] /ws/chat: нет reverse-WS узлов → VPS fallback")
        return await chat(message)

    # Выбираем лучший по tier
    tier_order = {"gpu": 0, "fast_cpu": 1, "slow_cpu": 2}
    worker_id  = min(
        available,
        key=lambda wid: tier_order.get(
            worker_stats.get(wid, {}).get("tier", "slow_cpu"), 3
        )
    )

    task_id = str(uuid.uuid4())
    ls = worker_stats[worker_id].get("layer_start", 0)
    le = worker_stats[worker_id].get("layer_end",   31)
    print(f"[Gateway] /ws/chat '{message[:50]}' → {worker_id[:16]} "
          f"(слои {ls}-{le})")

    try:
        result_text = await send_task_to_worker(worker_id, {
            "type":    "generate",
            "task_id": task_id,
            "message": message,
        })
        return {
            "response": result_text,
            "worker":   worker_id,
            "task_id":  task_id,
            "layers":   f"{ls}-{le}",
            "backend":  "airllm_ws",
        }
    except Exception as e:
        print(f"[Gateway] /ws/chat ошибка: {e} → VPS fallback")
        return await chat(message)


@app.get("/health")
def health():
    return {
        "status":           "ok",
        "model":            MODEL_NAME,
        "backend":          "airllm",
        "layers":           f"{LAYER_START}-{LAYER_END}",
        "total_layers":     total_layers,
        "queue_size":       len(pending_activations),
        "pipeline_ready":   get_coverage()["percent"] >= 80,
        "ws_port":          WS_PORT,
        "reverse_ws_nodes": len(reverse_ws_connections),
    }


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Noümind Pipeline Gateway  —  AirLLM Edition")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Layers: {LAYER_START}-{LAYER_END} ({total_layers} total)")
    print(f"  Port  : {PORT}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
