import bisect
import hashlib
import hmac
import html
import os
import re
from collections import defaultdict
from datetime import datetime
from difflib import get_close_matches
from threading import Lock
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
from flask import Flask, jsonify, request, send_from_directory


APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/Argentina/Buenos_Aires")
TARGETS_PATH = os.getenv("TARGETS_PATH", "data/targets_es.txt")
LEXICON_PATH = os.getenv("LEXICON_PATH", "data/lexicon.txt")
LEXICON_EMB_CACHE_PATH = os.getenv("LEXICON_EMB_CACHE_PATH", "data/lexicon_embeddings.npy")
TOP_K = int(os.getenv("TOP_K", "1000"))
MAX_WORD_LENGTH = int(os.getenv("MAX_WORD_LENGTH", "40"))
PUBLIC_REVEAL = os.getenv("PUBLIC_REVEAL", "true").lower() in {"1", "true", "yes"}
ALLOW_RANDOM_TARGET = os.getenv("ALLOW_RANDOM_TARGET", "true").lower() in {"1", "true", "yes"}
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "dev-secret-change-me")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

WORD_RE = re.compile(r"^[a-záéíóúüñ]+$", re.IGNORECASE)

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    if request.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "public, max-age=3600")
    else:
        response.headers.setdefault("Cache-Control", "no-store")
    origin = request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def normalize_word(word: str) -> str:
    return (word or "").strip().lower()


def load_wordlist(path: str) -> List[str]:
    if not os.path.exists(path):
        app.logger.warning("No existe %s", path)
        return []
    words: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = normalize_word(line)
            if word and " " not in word and len(word) <= MAX_WORD_LENGTH:
                words.append(word)
    return sorted(set(words))


def lexicon_fingerprint(words: List[str]) -> str:
    h = hashlib.sha256()
    for w in words:
        h.update(w.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def score_from_cos(cosine: float) -> float:
    return round(float(cosine) * 100.0, 1)


def deterministic_index(seed: int, n: int) -> int:
    digest = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def today_seed() -> Dict[str, object]:
    now = datetime.now(ZoneInfo(APP_TIMEZONE))
    return {"seed": int(now.strftime("%Y%m%d")), "date": now.strftime("%Y-%m-%d")}


def index_of(word: str) -> int:
    # LEXICON esta ordenado: busqueda binaria en vez de un dict de 708k entradas
    # (ese dict costaba ~50 MB de RAM que asi nos ahorramos).
    i = bisect.bisect_left(LEXICON, word)
    if i < len(LEXICON) and LEXICON[i] == word:
        return i
    return -1


def in_lexicon(word: str) -> bool:
    return index_of(word) >= 0


def is_valid_guess(word: str) -> bool:
    return bool(
        word
        and len(word) <= MAX_WORD_LENGTH
        and WORD_RE.match(word)
        and in_lexicon(word)
    )


def guess_suggestions(word: str) -> List[str]:
    if not word or len(word) < 3:
        return []
    bucket = SUGGEST_INDEX.get(word[0], [])
    candidates = [w for w in bucket if abs(len(w) - len(word)) <= 2]
    return get_close_matches(word, candidates, n=3, cutoff=0.78)


def sign_top_token(seed: int, word: str) -> str:
    payload = f"{seed}:{word}".encode("utf-8")
    return hmac.new(TOKEN_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def is_valid_top_token(seed: int, word: str, token: str) -> bool:
    return bool(token and hmac.compare_digest(sign_top_token(seed, word), token))


def load_embeddings(path: str, expected_rows: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontro {path}. Ejecuta: python tools/build_embeddings_cache.py"
        )
    emb = np.load(path, mmap_mode="r")
    if emb.shape[0] != expected_rows:
        raise RuntimeError(
            f"El cache {path} tiene {emb.shape[0]} filas, pero lexicon.txt tiene {expected_rows}."
        )
    return emb


app.logger.info("Cargando listas")
LEXICON = load_wordlist(LEXICON_PATH)
SUGGEST_INDEX: Dict[str, List[str]] = defaultdict(list)
for _word in LEXICON:
    SUGGEST_INDEX[_word[0]].append(_word)
TARGETS = [word for word in load_wordlist(TARGETS_PATH) if in_lexicon(word)]
TARGET_ROW = {word: i for i, word in enumerate(TARGETS)}

if not LEXICON:
    raise RuntimeError("El lexicon esta vacio. Revisa LEXICON_PATH.")
if not TARGETS:
    raise RuntimeError("Targets vacio o fuera del lexicon. Revisa TARGETS_PATH.")

app.logger.info("Cargando embeddings cacheados: %s", LEXICON_EMB_CACHE_PATH)
LEX_EMB_N = load_embeddings(LEXICON_EMB_CACHE_PATH, len(LEXICON))
EMBEDDING_DIM = int(LEX_EMB_N.shape[1])

META_PATH = f"{LEXICON_EMB_CACHE_PATH}.meta"
EXPECTED_FINGERPRINT = lexicon_fingerprint(LEXICON)
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as _f:
        STORED_FINGERPRINT = _f.read().strip()
    if STORED_FINGERPRINT != EXPECTED_FINGERPRINT:
        app.logger.warning(
            "El fingerprint del lexicon (%s) no coincide con el del cache (%s). "
            "El cache de embeddings puede estar desalineado: reconstruilo con "
            "tools/build_embeddings_cache.py",
            EXPECTED_FINGERPRINT,
            STORED_FINGERPRINT,
        )
else:
    app.logger.warning("No existe %s; no puedo validar la integridad del cache.", META_PATH)

# Top-K precalculado por target (tools/build_target_topk.py). Evita escanear los
# ~700k embeddings en cada cambio de target: asi el cache de 406 MB queda en disco
# (mmap) y solo tocamos filas sueltas -> RAM anonima del proceso < ~120 MB.
TOPK_PATH = os.getenv("TARGET_TOPK_PATH", "data/target_topk.npy")
if os.path.exists(TOPK_PATH):
    TARGET_TOPK = np.load(TOPK_PATH, mmap_mode="r")
    if TARGET_TOPK.shape[0] != len(TARGETS):
        raise RuntimeError(
            f"{TOPK_PATH} tiene {TARGET_TOPK.shape[0]} filas pero hay {len(TARGETS)} targets. "
            "Reconstruilo con tools/build_target_topk.py"
        )
    # El topk guarda indices al lexicon: si el lexicon o los targets cambiaron sin
    # regenerar el topk, los indices apuntarian a palabras equivocadas.
    TOPK_META_PATH = f"{TOPK_PATH}.meta"
    EXPECTED_TOPK_META = f"{lexicon_fingerprint(LEXICON)}:{lexicon_fingerprint(TARGETS)}:{TOP_K}"
    if os.path.exists(TOPK_META_PATH):
        with open(TOPK_META_PATH, "r", encoding="utf-8") as _f:
            stored_topk_meta = _f.read().strip()
        if stored_topk_meta != EXPECTED_TOPK_META:
            app.logger.warning(
                "El fingerprint del top-K precalculado (%s) no coincide con el esperado "
                "(%s). Puede estar desalineado: reconstruilo con tools/build_target_topk.py",
                stored_topk_meta,
                EXPECTED_TOPK_META,
            )
    else:
        app.logger.warning("No existe %s; no puedo validar la integridad del top-K.", TOPK_META_PATH)
else:
    raise FileNotFoundError(
        f"No se encontro {TOPK_PATH}. Ejecuta: py -3.11 tools/build_target_topk.py"
    )

STATE: Dict[str, object] = {
    "seed": None,
    "target_word": None,
    "target_index": None,
    "target_row": None,
    "target_emb_n": None,
    "topk_indices": None,
    "topk_sims": None,
    "rank_map": None,
    "thresholds": None,
}
STATE_LOCK = Lock()


def vector_for_word(word: str) -> np.ndarray:
    return np.asarray(LEX_EMB_N[index_of(word)], dtype=np.float32)


def build_topk_for_current_target() -> bool:
    target_emb_n = STATE["target_emb_n"]
    target_row = STATE["target_row"]
    if target_emb_n is None or target_row is None:
        return False

    # Indices ya ordenados por similitud (precalculados). Solo recalculamos las
    # ~1000 similitudes leyendo esas filas del cache (mmap): unos 600 KB tocados.
    topk_indices = np.asarray(TARGET_TOPK[int(target_row)], dtype=np.int64)
    rows = np.asarray(LEX_EMB_N[topk_indices], dtype=np.float32)
    topk_sims = rows @ np.asarray(target_emb_n, dtype=np.float32)

    rank_map = {int(idx): rank for rank, idx in enumerate(topk_indices, start=1)}
    thresholds = {
        "top1": score_from_cos(topk_sims[0]) if len(topk_sims) >= 1 else None,
        "top10": score_from_cos(topk_sims[9]) if len(topk_sims) >= 10 else None,
        "top1000": score_from_cos(topk_sims[999]) if len(topk_sims) >= 1000 else None,
    }

    STATE["topk_indices"] = topk_indices
    STATE["topk_sims"] = topk_sims
    STATE["rank_map"] = rank_map
    STATE["thresholds"] = thresholds
    return True


def initialize_target(seed: int) -> str:
    with STATE_LOCK:
        if STATE["seed"] == seed and STATE["target_word"]:
            return str(STATE["target_word"])

        target_row = deterministic_index(seed, len(TARGETS))
        target = TARGETS[target_row]
        target_emb_n = vector_for_word(target)

        STATE["seed"] = seed
        STATE["target_word"] = target
        STATE["target_index"] = index_of(target)
        STATE["target_row"] = target_row
        STATE["target_emb_n"] = target_emb_n
        build_topk_for_current_target()
        return target


def current_seed_from_request() -> int:
    data = request.get_json(silent=True) or {}
    seed = data.get("seed")
    if seed is not None:
        return int(seed)
    return int(today_seed()["seed"])


def game_info(seed: int, date: Optional[str] = None) -> Dict[str, object]:
    initialize_target(seed)
    return {
        "ok": True,
        "seed": seed,
        "date": date,
        "k": TOP_K,
        "thresholds": STATE["thresholds"],
    }


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/today")
def api_today():
    info = today_seed()
    return jsonify(game_info(int(info["seed"]), str(info["date"])))


@app.post("/new_target")
def new_target():
    data = request.get_json(silent=True) or {}
    seed = data.get("seed")
    random_mode = bool(data.get("random"))
    if random_mode and not ALLOW_RANDOM_TARGET:
        return jsonify(ok=False, error="La palabra aleatoria esta deshabilitada."), 403
    if seed is None:
        return jsonify(ok=False, error="Falta 'seed'."), 400
    return jsonify(game_info(int(seed)))


@app.post("/guess")
def guess():
    seed = current_seed_from_request()
    initialize_target(seed)

    data = request.get_json(silent=True) or {}
    word = normalize_word(data.get("word", ""))
    if not word:
        return jsonify(ok=False, error="Escribi una palabra."), 400
    if not is_valid_guess(word):
        return jsonify(
            ok=False,
            error="No encontre esa palabra en el diccionario del juego.",
            suggestions=guess_suggestions(word),
        ), 400

    target_word = str(STATE["target_word"])
    target_emb_n = STATE["target_emb_n"]
    guess_emb_n = vector_for_word(word)
    cosine = float(np.dot(guess_emb_n, target_emb_n))
    score = score_from_cos(cosine)

    correct = word == target_word
    rank: Optional[int] = None
    if correct:
        rank = 0
    elif STATE["rank_map"] is not None:
        rank = STATE["rank_map"].get(index_of(word))

    response = {
        "ok": True,
        "word": word,
        "score": score,
        "similarity_cosine": round(cosine, 6),
        "rank": rank,
        "k": TOP_K,
        "correct": correct,
    }
    if correct:
        response["top_token"] = sign_top_token(seed, word)
        response["nearest_url"] = f"/nearest?seed={seed}&word={word}&token={response['top_token']}"
    return jsonify(response)


@app.post("/reveal")
def reveal():
    if not PUBLIC_REVEAL:
        return jsonify(ok=False, error="Rendirse esta deshabilitado."), 403
    seed = current_seed_from_request()
    target_word = initialize_target(seed)
    token = sign_top_token(seed, target_word)
    return jsonify(
        ok=True,
        target_word=target_word,
        top_token=token,
        nearest_url=f"/nearest?seed={seed}&word={target_word}&token={token}",
    )


@app.get("/health")
def health():
    return jsonify(
        ok=True,
        mode="embeddings-only",
        embedding_cache=LEXICON_EMB_CACHE_PATH,
        embedding_dtype=str(LEX_EMB_N.dtype),
        embedding_dim=EMBEDDING_DIM,
        lexicon=len(LEXICON),
        targets=len(TARGETS),
        has_target=STATE["target_word"] is not None,
        top_k=TOP_K,
    )


@app.get("/nearest")
def nearest():
    seed = request.args.get("seed", type=int)
    guess = normalize_word(request.args.get("word") or request.args.get("guess") or "")
    token = request.args.get("token", "")
    if seed is None or not guess or not is_valid_top_token(seed, guess, token):
        return "Top 1000 disponible solo al ganar o rendirse.", 403

    initialize_target(seed)
    if STATE["topk_indices"] is None or STATE["topk_sims"] is None:
        return "Target no inicializado.", 400

    rows = []
    target_word = str(STATE["target_word"])
    for rank, idx in enumerate(STATE["topk_indices"], start=1):
        word = LEXICON[int(idx)]
        sim = float(STATE["topk_sims"][rank - 1])
        score = score_from_cos(sim)
        cls = " class='guess'" if word == guess else ""
        rows.append(
            f"<tr{cls}><td>{rank}</td><td>{html.escape(word)}</td>"
            f"<td>{score:.1f}</td><td>{sim:.6f}</td></tr>"
        )

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Top {TOP_K} - Semantle ES</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial;background:#f7f7f4;color:#1f2933;margin:0}}
.wrap{{max-width:920px;margin:0 auto;padding:24px 16px 40px}}
h1{{font-size:24px;margin:0 0 8px}}
.meta{{color:#5f665f;margin-bottom:14px}}
.pill{{display:inline-block;padding:3px 8px;border-radius:999px;background:#e7e5dc;font-size:12px}}
table{{width:100%;border-collapse:collapse;background:white;border:1px solid #dfddd3}}
th,td{{padding:10px 12px;border-bottom:1px solid #efeee8;text-align:left;font-size:14px}}
thead th{{background:#eeece2;color:#3f463f;font-weight:700}}
tr.guess td{{background:#fff3bf}}
</style>
</head>
<body>
<div class="wrap">
<h1>Top {TOP_K} palabras mas cercanas</h1>
<div class="meta">Target: <span class="pill">{html.escape(target_word)}</span> &nbsp; Intento: <span class="pill">{html.escape(guess)}</span></div>
<table><thead><tr><th>#</th><th>Palabra</th><th>Score</th><th>Cosine</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</div>
</body>
</html>"""


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=debug)
