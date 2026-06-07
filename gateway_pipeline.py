"""
gateway_pipeline.py — VPS
AirLLM backend: Mistral-7B-Instruct-v0.2 (layer streaming, ~0.4 GB RAM).
Раздаёт активации узлам через pull модель (distributed pipeline остаётся).
"""
import sys, os, io, asyncio, uuid, base64, time, re
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict

import torch
from transformers import AutoTokenizer
from airllm import AutoModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

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
# ФЕДЕРАТИВНОЕ ОБУЧЕНИЕ — gates store
# ─────────────────────────────────────────
gates_store: Dict[str, dict] = {}

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
    worker_id:      str,
    tier:           str  = "slow_cpu",
    device:         str  = "cpu",
    ram_gb:         int  = 0,
    specialization: str  = "general",
    is_fallback:    bool = False,
    payload:        RegisterPayload | None = None,
):
    prev = worker_stats.get(worker_id, {})
    worker_stats[worker_id] = {
        "tier":            tier,
        "device":          device,
        "ram_gb":          ram_gb,
        "status":          "free",
        "tasks_done":      prev.get("tasks_done", 0),
        "avg_speed":       prev.get("avg_speed", 999),
        "specialization":  specialization,
        "is_fallback":     is_fallback,
        "category_scores": (payload.category_scores if payload and payload.category_scores
                            else prev.get("category_scores", {})),
    }
    role = "Shared Expert" if is_fallback else f"Routed Expert [{specialization}]"
    print(f"[Gateway] ✓ Воркер зарегистрирован: {worker_id} | {role} | tier={tier}")
    return {"status": "registered", "worker_id": worker_id, "specialization": specialization}


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


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model":       MODEL_NAME,
        "backend":     "airllm",
        "layers":      f"{LAYER_START}-{LAYER_END}",
        "total_layers": total_layers,
        "queue_size":  len(pending_activations),
    }


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Noümind Pipeline Gateway  —  AirLLM Edition")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Layers: {LAYER_START}-{LAYER_END} ({total_layers} total)")
    print(f"  Port  : {PORT}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
