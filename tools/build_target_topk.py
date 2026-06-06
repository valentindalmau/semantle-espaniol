"""Precalcula el top-1000 de palabras mas cercanas para cada target.

Con esto el servidor NO necesita escanear los 708k embeddings en cada cambio de
target (lo que forzaba tener los ~406 MB residentes). En runtime solo lee la fila
precalculada del target del dia (1000 indices) y, por mmap, unas pocas filas del
cache. La RAM anonima del proceso queda por debajo de ~120 MB sin perder calidad:
los embeddings siguen siendo los full 300-d float16.

Salida:
  data/target_topk.npy       (n_targets x TOP_K) uint32, indices al lexicon,
                             ordenados por similitud desc, alineados al orden
                             ordenado de targets_es.txt (mismo que usa app.py).
  data/target_topk.npy.meta  fingerprint(lexicon):fingerprint(targets):TOP_K

Correr cada vez que cambie el lexicon, el cache o los targets:
  py -3.11 tools/build_target_topk.py
"""
from __future__ import annotations

import argparse
import hashlib
import time
import unicodedata
from pathlib import Path
from typing import List

import numpy as np

MAX_WORD_LENGTH = 40


def nfc_lower(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip().lower())


def load_wordlist(path: Path) -> List[str]:
    words: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = nfc_lower(line)
            if w and " " not in w and len(w) <= MAX_WORD_LENGTH:
                words.append(w)
    return sorted(set(words))


def fingerprint(words: List[str]) -> str:
    h = hashlib.sha256()
    for w in words:
        h.update(w.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precalcular top-K por target")
    parser.add_argument("--lexicon", default="data/lexicon.txt")
    parser.add_argument("--targets", default="data/targets_es.txt")
    parser.add_argument("--cache", default="data/lexicon_embeddings.npy")
    parser.add_argument("--out", default="data/target_topk.npy")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--target-batch", type=int, default=128)
    parser.add_argument("--row-chunk", type=int, default=100000)
    args = parser.parse_args()

    lexicon = load_wordlist(Path(args.lexicon))
    index = {w: i for i, w in enumerate(lexicon)}
    targets = [w for w in load_wordlist(Path(args.targets)) if w in index]

    emb = np.load(args.cache, mmap_mode="r")
    n, dim = emb.shape
    if n != len(lexicon):
        raise RuntimeError(f"Cache tiene {n} filas pero lexicon {len(lexicon)}.")

    top_k = args.top_k
    nt = len(targets)
    target_rows = np.array([index[w] for w in targets], dtype=np.int64)
    tvecs = np.asarray(emb[target_rows], dtype=np.float32)  # nt x dim

    out = np.empty((nt, top_k), dtype=np.uint32)
    t0 = time.time()
    B, CH = args.target_batch, args.row_chunk
    for b0 in range(0, nt, B):
        tb = tvecs[b0:b0 + B]                       # b x dim
        sims = np.empty((tb.shape[0], n), dtype=np.float32)
        for c0 in range(0, n, CH):
            c1 = min(c0 + CH, n)
            chunk = np.asarray(emb[c0:c1], dtype=np.float32)   # ch x dim
            sims[:, c0:c1] = tb @ chunk.T
        for j in range(tb.shape[0]):
            ti = b0 + j
            s = sims[j]
            cand = np.argpartition(-s, top_k)[:top_k + 1]
            cand = cand[np.argsort(-s[cand])]
            cand = cand[cand != target_rows[ti]][:top_k]
            out[ti] = cand.astype(np.uint32)
        print(f"{min(b0 + B, nt):,}/{nt:,}", flush=True)

    np.save(args.out, out)
    meta = f"{fingerprint(lexicon)}:{fingerprint(targets)}:{top_k}"
    Path(f"{args.out}.meta").write_text(meta, encoding="utf-8")
    print(f"OK -> {args.out} ({nt:,} x {top_k}) en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
