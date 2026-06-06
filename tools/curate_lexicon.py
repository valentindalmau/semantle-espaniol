from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Iterable, Set

from wordfreq import zipf_frequency


WORD_RE = re.compile(r"^[a-záéíóúüñ]+$", re.IGNORECASE)
VOWEL_RE = re.compile(r"[aeiouáéíóúü]", re.IGNORECASE)
TRIPLE_RE = re.compile(r"(.)\1\1", re.IGNORECASE)
BAD_SEQUENCES = ("www", "http", "nbsp", "quot", "amp")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip().lower())


def iter_words(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = normalize(line)
            if word:
                yield word


def is_shape_ok(word: str, min_len: int, max_len: int) -> bool:
    if not (min_len <= len(word) <= max_len):
        return False
    if not WORD_RE.match(word):
        return False
    if not VOWEL_RE.search(word):
        return False
    if TRIPLE_RE.search(word):
        return False
    if any(seq in word for seq in BAD_SEQUENCES):
        return False
    return True


def is_common_enough(word: str, min_zipf: float) -> bool:
    return zipf_frequency(word, "es") >= min_zipf


def curate(
    source: Path,
    targets: Path,
    min_zipf: float,
    min_len: int,
    max_len: int,
    keep_targets: bool,
) -> Set[str]:
    kept: Set[str] = set()

    for word in iter_words(source):
        if is_shape_ok(word, min_len, max_len) and is_common_enough(word, min_zipf):
            kept.add(word)

    if keep_targets and targets.exists():
        for word in iter_words(targets):
            if is_shape_ok(word, min_len, max_len):
                kept.add(word)

    return kept


def write_words(path: Path, words: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for word in sorted(words):
            f.write(word + "\n")


def remove_cache(cache_path: Path) -> None:
    for path in (cache_path, Path(f"{cache_path}.meta")):
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Curar lexicon.txt para Semantle ES")
    parser.add_argument("--source", default="data/lexicon.txt")
    parser.add_argument("--input", default=None, help="Archivo desde el que leer. Por defecto usa --source.")
    parser.add_argument("--targets", default="data/targets_es.txt")
    parser.add_argument("--backup", default="data/lexicon.raw.txt")
    parser.add_argument("--targets-backup", default="data/targets_es.raw.txt")
    parser.add_argument("--cache", default="data/lexicon_embeddings.npy")
    parser.add_argument("--min-zipf", type=float, default=2.5)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--keep-targets", action="store_true", help="Mantener targets validos aunque no pasen frecuencia.")
    parser.add_argument("--no-curate-targets", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    targets = Path(args.targets)
    backup = Path(args.backup)
    targets_backup = Path(args.targets_backup)
    cache = Path(args.cache)

    input_path = Path(args.input) if args.input else source

    if not source.exists():
        raise FileNotFoundError(source)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if not backup.exists():
        shutil.copy2(source, backup)
        print(f"Backup creado: {backup}")
    else:
        print(f"Backup existente: {backup}")

    original_count = sum(1 for _ in iter_words(input_path))
    curated = curate(
        source=input_path,
        targets=targets,
        min_zipf=args.min_zipf,
        min_len=args.min_len,
        max_len=args.max_len,
        keep_targets=args.keep_targets,
    )
    write_words(source, curated)

    if targets.exists() and not args.no_curate_targets:
        if not targets_backup.exists():
            shutil.copy2(targets, targets_backup)
            print(f"Backup de targets creado: {targets_backup}")
        else:
            print(f"Backup de targets existente: {targets_backup}")

        original_targets = [w for w in iter_words(targets)]
        curated_targets = [w for w in original_targets if w in curated]
        write_words(targets, curated_targets)
        print(f"Targets originales: {len(original_targets):,}")
        print(f"Targets curados:    {len(curated_targets):,}")

    remove_cache(cache)

    print(f"Original: {original_count:,} entradas")
    print(f"Curado:   {len(curated):,} entradas")
    print(f"Cache removido: {cache} (+ .meta)")


if __name__ == "__main__":
    main()
