from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import fasttext
import numpy as np


# Debe coincidir con MAX_WORD_LENGTH y load_wordlist de app.py para que el orden
# de las filas del cache sea identico al LEXICON que usa el servidor.
MAX_WORD_LENGTH = 40


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def lexicon_fingerprint(words: list[str]) -> str:
    h = hashlib.sha256()
    for w in words:
        h.update(w.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def load_words(path: Path) -> list[str]:
    words: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word and " " not in word and len(word) <= MAX_WORD_LENGTH:
                words.append(word)
    return sorted(set(words))


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir cache de embeddings para Semantle ES")
    parser.add_argument("--model", default="data/cc.es.300.bin")
    parser.add_argument("--lexicon", default="data/lexicon.txt")
    parser.add_argument("--out", default="data/lexicon_embeddings.npy")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=20000)
    args = parser.parse_args()

    model_path = Path(args.model)
    lexicon_path = Path(args.lexicon)
    out_path = Path(args.out)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not lexicon_path.exists():
        raise FileNotFoundError(lexicon_path)

    words = load_words(lexicon_path)
    model = fasttext.load_model(str(model_path))
    dim = model.get_dimension()
    dtype = np.float16 if args.dtype == "float16" else np.float32

    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(len(words), dim))

    for start in range(0, len(words), args.batch_size):
        batch = words[start:start + args.batch_size]
        vectors = np.vstack([model.get_word_vector(word) for word in batch]).astype(np.float32)
        emb[start:start + len(batch)] = normalize_matrix(vectors).astype(dtype)
        print(f"{start + len(batch):,}/{len(words):,}", flush=True)

    meta_path = Path(f"{out_path}.meta")
    meta_path.write_text(lexicon_fingerprint(words), encoding="utf-8")
    print(f"Cache escrito: {out_path} ({emb.shape[0]:,} x {emb.shape[1]}, {args.dtype})")


if __name__ == "__main__":
    main()
