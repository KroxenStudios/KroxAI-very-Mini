"""KroxAI HTTP server using FastAPI with lightweight chat memory and decoding params.
Endpoints:
- GET /health
- POST /chat { text: str, conversation_id?: str, temperature?: float, top_p?: float, max_new_tokens?: int }
"""
import os
import time
import uuid
import json
import sys
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, Deque, Dict, List, Tuple
from collections import deque
import sys
from pathlib import Path

# Ensure parent directory is on sys.path when running as a script
_this_file = Path(__file__).resolve()
_pkg_root = _this_file.parent
_ws_root = _pkg_root.parent
if str(_ws_root) not in sys.path:
    sys.path.insert(0, str(_ws_root))

# Robust imports: prefer relative (package mode), fallback to absolute or local (script mode)
try:
    from .torch_chat import KroxAI
except Exception:
    # If running as a script (python kroxai/server.py), ensure parent dir is on sys.path
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(this_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        from kroxai.torch_chat import KroxAI  # type: ignore
    except Exception:
        try:
            from torch_chat import KroxAI  # type: ignore
        except Exception:
            raise

try:
    from .ce_reranker import get_ce_reranker
except Exception:
    try:
        from kroxai.ce_reranker import get_ce_reranker  # type: ignore
    except Exception:
        # Optional fallback if reranker module is unavailable
        def get_ce_reranker(model_name: str):
            class _Dummy:
                ok = False
                def rerank(self, q, pairs, k=3):
                    return [(p, 0.0) for p in pairs]
                def info(self):
                    return {"ok": False}
            return _Dummy()

# Optional BM25 (MIT: rank_bm25). If missing, use simple keyword scoring.
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

app = FastAPI(title="KroxAI Server")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Show the full app UI from templates/index.html
    return templates.TemplateResponse("index.html", {"request": request})

# Mount static and templates (use workspace-level defaults if present)
TEMPLATES_DIR = os.environ.get("KROXAI_TEMPLATES_DIR", os.path.join(_ws_root, "templates"))
STATIC_DIR = os.environ.get("KROXAI_STATIC_DIR", os.path.join(_ws_root, "static"))
try:
    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
except Exception:
    pass
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# CORS (config via env KROXAI_CORS_ORIGINS, comma separated or *)
origins = os.environ.get("KROXAI_CORS_ORIGINS", "*")
origin_list = [o.strip() for o in origins.split(",")] if origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
agent = KroxAI()
MEMORY: Dict[str, Deque[Tuple[str, str]]] = {}
# Configurable history turns and max context size
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

HISTORY_TURNS = _env_int("KROXAI_MEMORY_TURNS", 6)  # keep last N turns
MAX_CONTEXT_CHARS = _env_int("KROXAI_MAX_CONTEXT_CHARS", 8000)

# Optional API key
API_KEY = os.environ.get("KROXAI_API_KEY")

def _check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# RAG store
RAG_ENABLED = os.environ.get("KROXAI_RAG", "1") != "0"
RAG_DOCS: List[Dict[str, str]] = []  # each: {id, text}
RAG_TOKENS: List[List[str]] = []
RAG_INDEX = None
CE_RERANK = os.environ.get("KROXAI_CE_RERANK", "0") == "1"
CE_MODEL = os.environ.get("KROXAI_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RAG_PATH = os.environ.get("KROXAI_RAG_PATH", os.path.join(os.getcwd(), "rag_store.jsonl"))

def _simple_tokenize(s: str) -> List[str]:
    return [t.lower() for t in s.split()]

def _rebuild_rag_index():
    global RAG_INDEX
    if not RAG_DOCS:
        RAG_INDEX = None
        return
    if HAS_BM25:
        RAG_INDEX = BM25Okapi(RAG_TOKENS)
    else:
        RAG_INDEX = None  # keyword fallback

def _save_rag(path: Optional[str] = None) -> str:
    p = path or RAG_PATH
    try:
        with open(p, "w", encoding="utf-8") as f:
            for d in RAG_DOCS:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return p

def _load_rag(path: Optional[str] = None, *, replace: bool = True) -> int:
    p = path or RAG_PATH
    count = 0
    docs: List[Dict[str, str]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if isinstance(d, dict) and "text" in d:
                        docs.append({"id": str(d.get("id") or str(len(docs)+1)), "text": str(d["text"])})
                except Exception:
                    continue
    except Exception:
        docs = []
    if docs:
        if replace:
            RAG_DOCS.clear()
            RAG_TOKENS.clear()
        for d in docs:
            RAG_DOCS.append(d)
            RAG_TOKENS.append(_simple_tokenize(d["text"]))
        _rebuild_rag_index()
        count = len(docs)
    return count

def _rag_search(query: str, k: int = 3) -> List[Dict[str, str]]:
    if not RAG_DOCS:
        return []
    toks = _simple_tokenize(query)
    hits_idx: List[Tuple[int, float]] = []
    if HAS_BM25 and RAG_INDEX is not None:
        try:
            scores = RAG_INDEX.get_scores(toks)
            hits_idx = [(i, float(scores[i])) for i in range(len(scores))]
        except Exception:
            hits_idx = []
    if not hits_idx:
        # keyword fallback scoring
        qset = set(toks)
        for i, toks_doc in enumerate(RAG_TOKENS):
            hits_idx.append((i, float(len(qset.intersection(toks_doc)))))
    # initial top-k by base score
    hits_idx.sort(key=lambda x: x[1], reverse=True)
    top_hits = hits_idx[: max(k, 5)]  # take a bit more for reranker headroom

    # Optional CE rerank
    if CE_RERANK and top_hits:
        try:
            ce = get_ce_reranker(CE_MODEL)
            if getattr(ce, "ok", False):
                pairs = [ (RAG_DOCS[i], sc) for (i, sc) in top_hits ]
                reranked = ce.rerank(query, pairs, k=k)
                # Map back to docs
                out_docs: List[Dict[str, str]] = []
                for doc, _ in reranked:
                    # doc is already the original dict
                    if isinstance(doc, dict) and "text" in doc:
                        out_docs.append(doc)
                if out_docs:
                    return out_docs[:k]
        except Exception:
            pass

    # Fallback: return top base-score docs
    out = [RAG_DOCS[i] for (i, sc) in top_hits if sc > 0][:k]
    return out

class ChatIn(BaseModel):
    text: str
    conversation_id: Optional[str] = None
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 0.95
    max_new_tokens: Optional[int] = 160

class ChatOut(BaseModel):
    reply: str

@app.get("/health")
def health(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    return {"status": "ok"}

@app.post("/chat", response_model=ChatOut)
def chat(inp: ChatIn, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    # Initialize memory
    conv_id = inp.conversation_id or "default"
    dq = MEMORY.get(conv_id)
    if dq is None:
        dq = deque(maxlen=HISTORY_TURNS)
        MEMORY[conv_id] = dq

    # Append user message
    dq.append(("user", inp.text))

    # Build prompt from history + optional RAG
    context_blocks: List[str] = []
    if RAG_ENABLED:
        try:
            topk = int(os.environ.get("KROXAI_RAG_TOPK", "3"))
        except Exception:
            topk = 3
        hits = _rag_search(inp.text, k=topk)
        if hits:
            context_blocks.append("\n\n".join([f"[Doc {i+1}]\n" + h["text"] for i, h in enumerate(hits)]))
    history_text = "\n".join([f"{r.upper()}: {t}" for r, t in dq])
    context_text = ("\n\nCONTEXT:\n" + "\n\n".join(context_blocks)) if context_blocks else ""
    prompt = f"You are a helpful assistant. Use CONTEXT if relevant to answer concisely. {context_text}\n\nDIALOGUE:\n{history_text}\nASSISTANT:"
    # Enforce max context size
    if len(prompt) > MAX_CONTEXT_CHARS:
        prompt = prompt[-MAX_CONTEXT_CHARS:]

    # Let the agent respond (agent internally may ignore extra params; we can extend later)
    # If transformers are available in the shim, use generate with decoding params; else handle_user_message
    try:
        raw_reply = agent.generate(
            prompt,
            temperature=inp.temperature or 0.9,
            top_p=inp.top_p or 0.95,
            max_new_tokens=inp.max_new_tokens or 160,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
    except Exception:
        raw_reply = agent.handle_user_message(prompt)

    # Post-process: take only the new assistant content after 'ASSISTANT:' if present
    reply = raw_reply
    marker = "ASSISTANT:"
    if marker in raw_reply:
        reply = raw_reply.split(marker, 1)[-1].strip()

    # Basic noise filter: overly long repeated chars or very low alpha ratio
    try:
        import re
        if re.search(r"(.)\1{8,}", reply):
            reply = "Ich konnte darauf keine sinnvolle Antwort generieren. Kannst du deine Frage präzisieren?"
        alpha = sum(c.isalpha() for c in reply) + 1
        ratio = alpha / (len(reply) + 1)
        if ratio < 0.2:
            reply = "Ich konnte darauf keine sinnvolle Antwort generieren. Kannst du deine Frage präzisieren?"
    except Exception:
        pass

    # Append assistant reply to memory
    dq.append(("assistant", reply))

    return ChatOut(reply=reply)

@app.get("/app", response_class=HTMLResponse)
def app_page(request: Request):
    # Render chat UI template
    return templates.TemplateResponse("chat.html", {"request": request})

# RAG endpoints
class RagAddIn(BaseModel):
    id: Optional[str] = None
    text: str

class RagSearchIn(BaseModel):
    query: str
    k: Optional[int] = 3

class RagBackupIn(BaseModel):
    path: Optional[str] = None

class RagRestoreIn(BaseModel):
    path: Optional[str] = None
    mode: Optional[str] = "replace"  # replace|merge

@app.post("/rag/add")
def rag_add(inp: RagAddIn, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    doc_id = inp.id or str(len(RAG_DOCS) + 1)
    RAG_DOCS.append({"id": doc_id, "text": inp.text})
    RAG_TOKENS.append(_simple_tokenize(inp.text))
    _rebuild_rag_index()
    return {"ok": True, "count": len(RAG_DOCS)}

@app.post("/rag/search")
def rag_search(inp: RagSearchIn, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    hits = _rag_search(inp.query, inp.k or 3)
    return {"hits": hits}

@app.get("/rag/info")
def rag_info(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    # CE reranker status
    ce_status = {"enabled": CE_RERANK}
    if CE_RERANK:
        try:
            ce = get_ce_reranker(CE_MODEL)
            ce_status.update(ce.info())
        except Exception:
            ce_status.update({"ok": False})
    return {"docs": len(RAG_DOCS), "bm25": HAS_BM25, "reranker": ce_status, "path": RAG_PATH}

@app.post("/rag/backup")
def rag_backup(inp: RagBackupIn, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    p = _save_rag(inp.path)
    return {"ok": True, "path": p, "docs": len(RAG_DOCS)}

@app.post("/rag/restore")
def rag_restore(inp: RagRestoreIn, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    replace = (inp.mode or "replace").lower().strip() != "merge"
    n = _load_rag(inp.path, replace=replace)
    return {"ok": True, "loaded": n, "docs": len(RAG_DOCS)}

@app.get("/rag/export")
def rag_export(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    return {"docs": RAG_DOCS}

@app.post("/rag/import")
def rag_import(body: Dict, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    docs = body.get("docs") if isinstance(body, dict) else None
    if not isinstance(docs, list):
        return {"ok": False, "error": "docs must be a list"}
    for d in docs:
        if isinstance(d, dict) and "text" in d:
            RAG_DOCS.append({"id": str(d.get("id") or str(len(RAG_DOCS)+1)), "text": str(d["text"])})
            RAG_TOKENS.append(_simple_tokenize(str(d["text"])))
    _rebuild_rag_index()
    return {"ok": True, "docs": len(RAG_DOCS)}

@app.post("/rag/clear")
def rag_clear(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    RAG_DOCS.clear()
    RAG_TOKENS.clear()
    _rebuild_rag_index()
    return {"ok": True, "docs": 0}

# -------- Rate limiting, metrics, and structured logs --------
RATE_LIMIT_CFG = os.environ.get("KROXAI_RATE_LIMIT", "")  # e.g. "60/60" = 60 req per 60s per IP
def _parse_rate_limit(cfg: str) -> Tuple[int, int]:
    try:
        if not cfg:
            return (0, 0)
        if "/" in cfg:
            n, w = cfg.split("/", 1)
            return (max(0, int(n)), max(1, int(w)))
        # fallback: just number per 60s
        return (max(0, int(cfg)), 60)
    except Exception:
        return (0, 0)

RL_MAX, RL_WIN = _parse_rate_limit(RATE_LIMIT_CFG)
from collections import defaultdict
from datetime import datetime

_rl_buckets: Dict[str, Deque[float]] = defaultdict(lambda: deque())
_metrics = {
    "requests_total": 0,
    "path_counts": {},
    "rate_limited_total": 0,
    "durations_ms_sum": 0.0,
}

def _metrics_inc_path(path: str):
    d = _metrics["path_counts"]
    d[path] = int(d.get(path, 0)) + 1

def _rl_allow(ip: str, now: float) -> bool:
    if RL_MAX <= 0:
        return True
    q = _rl_buckets[ip]
    # drop old
    while q and now - q[0] > RL_WIN:
        q.popleft()
    if len(q) >= RL_MAX:
        return False
    q.append(now)
    return True

def _client_ip(request: Request) -> str:
    try:
        ip = request.headers.get("X-Forwarded-For")
        if ip:
            return ip.split(",")[0].strip()
        return request.client.host if request.client else "-"
    except Exception:
        return "-"

@app.middleware("http")
async def _obs_mw(request: Request, call_next):
    start = time.perf_counter()
    rid = uuid.uuid4().hex
    ip = _client_ip(request)
    path = request.url.path
    # Rate limit only chat/rag endpoints
    if path.startswith("/chat") or path.startswith("/rag"):
        if not _rl_allow(ip, time.time()):
            _metrics["rate_limited_total"] += 1
            data = {"ts": datetime.utcnow().isoformat()+"Z", "rid": rid, "ip": ip, "path": path, "status": 429, "msg": "rate_limited"}
            print(json.dumps(data, ensure_ascii=False))
            return Response(content=json.dumps({"error": "rate_limited", "request_id": rid}), status_code=429, media_type="application/json")
    try:
        resp = await call_next(request)
    except Exception as e:
        dur = (time.perf_counter() - start) * 1000.0
        _metrics["requests_total"] += 1
        _metrics_inc_path(path)
        _metrics["durations_ms_sum"] += dur
        data = {"ts": datetime.utcnow().isoformat()+"Z", "rid": rid, "ip": ip, "path": path, "status": 500, "ms": round(dur,2), "error": str(e)}
        print(json.dumps(data, ensure_ascii=False))
        return Response(content=json.dumps({"error": "server_error", "request_id": rid}), status_code=500, media_type="application/json")

    dur = (time.perf_counter() - start) * 1000.0
    _metrics["requests_total"] += 1
    _metrics_inc_path(path)
    _metrics["durations_ms_sum"] += dur
    # Add headers
    try:
        resp.headers["X-Request-ID"] = rid
        resp.headers["X-Response-Time"] = f"{dur:.2f}ms"
    except Exception:
        pass
    # Structured log
    data = {"ts": datetime.utcnow().isoformat()+"Z", "rid": rid, "ip": ip, "path": path, "status": resp.status_code, "ms": round(dur,2)}
    print(json.dumps(data, ensure_ascii=False))
    return resp

@app.get("/metrics")
def metrics():
    # Minimal Prometheus format
    lines: List[str] = []
    lines.append(f"kroxai_requests_total {int(_metrics['requests_total'])}")
    for p, c in sorted(_metrics["path_counts"].items()):
        lines.append(f"kroxai_path_requests_total{{path=\"{p}\"}} {int(c)}")
    lines.append(f"kroxai_rate_limited_total {int(_metrics['rate_limited_total'])}")
    lines.append(f"kroxai_durations_ms_sum {_metrics['durations_ms_sum']:.2f}")
    return Response(content="\n".join(lines)+"\n", media_type="text/plain; version=0.0.4")

# ----------------- Minimal endpoints for index.html UI -----------------
class SendIn(BaseModel):
    text: str
    sid: Optional[str] = None
    combine: Optional[bool] = False
    n: Optional[int] = None
    method: Optional[str] = None
    search: Optional[bool] = None
    rag: Optional[bool] = None
    trace: Optional[bool] = None
    attachments: Optional[List[str]] = None
    lang: Optional[str] = None

@app.post("/api/send")
def api_send(inp: SendIn, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    # Conversation memory by session id
    sid = inp.sid or "default"
    dq = MEMORY.get(sid)
    if dq is None:
        dq = deque(maxlen=HISTORY_TURNS)
        MEMORY[sid] = dq
    dq.append(("user", inp.text))

    # Optional RAG
    use_rag = bool(inp.rag) if inp.rag is not None else RAG_ENABLED
    context_blocks: List[str] = []
    if use_rag:
        try:
            topk = int(os.environ.get("KROXAI_RAG_TOPK", "3"))
        except Exception:
            topk = 3
        hits = _rag_search(inp.text, k=topk)
        if hits:
            context_blocks.append("\n\n".join([f"[Doc {i+1}]\n" + h["text"] for i, h in enumerate(hits)]))
    history_text = "\n".join([f"{r.upper()}: {t}" for r, t in dq])
    context_text = ("\n\nCONTEXT:\n" + "\n\n".join(context_blocks)) if context_blocks else ""
    prompt = f"You are a helpful assistant. Use CONTEXT if relevant to answer concisely. {context_text}\n\nDIALOGUE:\n{history_text}\nASSISTANT:"
    if len(prompt) > MAX_CONTEXT_CHARS:
        prompt = prompt[-MAX_CONTEXT_CHARS:]

    try:
        raw_reply = agent.generate(prompt)
    except Exception:
        raw_reply = agent.handle_user_message(prompt)
    reply = raw_reply.split("ASSISTANT:", 1)[-1].strip() if "ASSISTANT:" in raw_reply else raw_reply
    dq.append(("assistant", reply))

    # Minimal response to satisfy UI (no evidence/traces for now)
    return {"ok": True, "text": reply}

@app.get("/info")
def info(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    # Minimal system info for status bar
    try:
        device = "cuda" if (os.environ.get("KROXAI_DEVICE") == "cuda") else "cpu"
    except Exception:
        device = "cpu"
    return {
        "device": device,
        "preset": "shim",
        "tokenizer": {"type": "simple"},
        "checkpoint": None,
    }

class NTpIn(BaseModel):
    prefix: str

@app.post("/api/next_token")
def next_token(inp: NTpIn, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    p = (inp.prefix or "").strip()
    # Tiny heuristic suggestion: add a space and a period if sentence seems complete
    suggestion = ""
    try:
        if p and not p.endswith(" "):
            suggestion = " "
        if p and p[-1].isalnum():
            suggestion += "."
    except Exception:
        suggestion = ""
    return {"ok": True, "suggestion": suggestion}

class ClearIn(BaseModel):
    sid: Optional[str] = None

@app.post("/memory/clear")
def memory_clear(inp: ClearIn, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    sid = inp.sid or "default"
    if sid in MEMORY:
        try:
            del MEMORY[sid]
        except Exception:
            MEMORY[sid].clear()
    return {"ok": True}

def main():
    # Default to the requested LAN address unless overridden by env vars
    host = os.environ.get("KROXAI_HOST", "192.168.178.116")
    port = int(os.environ.get("KROXAI_PORT", "5000"))
    # Ensure CORS origins include the host if using default
    try:
        origins_env = os.environ.get("KROXAI_CORS_ORIGINS")
        if not origins_env:
            os.environ["KROXAI_CORS_ORIGINS"] = f"http://{host}:{port}"
    except Exception:
        pass
    uvicorn.run("kroxai.server:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
