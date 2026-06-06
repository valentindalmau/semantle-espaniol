"""Reconstruye data/lexicon.txt con cobertura completa y acentos correctos.

Problema que resuelve:
  - Cobertura: el lexicon viejo se armo por frecuencia, asi que faltaban formas
    flexionadas validas (mechado, mechados, mechadas estaban afuera; mechada no).
  - Acentos: convivian duplicados sin tilde y con tilde (accion + accion[con
    tilde]). Las versiones sin tilde son basura que ensucia los embeddings.

Como lo resuelve:
  1. Expande TODO el diccionario hunspell del espanol (spylls) aplicando sufijos,
     prefijos, cross-product y continuaciones -> todas las formas validas y bien
     acentuadas (~650k).
  2. Une con el lexicon actual para no perder nombres propios / prestamos que la
     gente igual quiere adivinar (espana, francia, messi, internet...).
  3. Dedup de acentos: descarta una forma SIN tilde solo si no es una palabra
     valida por si misma y existe su variante acentuada (saca 'accion' porque
     existe 'accion'->'accion' acentuada, pero conserva 'publico' que es una
     conjugacion valida distinta de 'publico' acentuado).

Tras correr esto hay que regenerar el cache:  python tools/build_embeddings_cache.py
"""
from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import requests
from spylls.hunspell import Dictionary

DIC_BASE_URL = "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/es/index"
WORD_RE = re.compile(r"^[a-záéíóúüñ]+$")
ACCENTS = "áéíóúüñ"


def nfc_lower(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip().lower())


def strip_accents(text: str) -> str:
    # quita tildes/dieresis pero conserva la enie
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch) == "Mn" and ch != "̃":  # 0303 = tilde de la enie
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def ensure_dictionary(base: Path) -> Dictionary:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("dic", "aff"):
        target = base.with_suffix(f".{ext}")
        if not target.exists():
            print(f"Descargando diccionario es ({ext})...")
            resp = requests.get(f"{DIC_BASE_URL}.{ext}", timeout=60)
            resp.raise_for_status()
            target.write_text(resp.text, encoding="utf-8")
    return Dictionary.from_files(str(base))


def _apply_suffix(word: str, sfx) -> str | None:
    if sfx.strip and not word.endswith(sfx.strip):
        return None
    if sfx.cond_regexp and not sfx.cond_regexp.search(word):
        return None
    base = word[: len(word) - len(sfx.strip)] if sfx.strip else word
    return base + sfx.add


def _apply_prefix(word: str, pfx) -> str | None:
    if pfx.strip and not word.startswith(pfx.strip):
        return None
    if pfx.cond_regexp and not pfx.cond_regexp.search(word):
        return None
    base = word[len(pfx.strip):] if pfx.strip else word
    return pfx.add + base


def expand_stem(stem: str, flags: Set[str], aff, depth: int = 2) -> Set[str]:
    out = {stem}
    frontier = [(stem, flags, depth)]
    while frontier:
        w, fl, dep = frontier.pop()
        if dep <= 0:
            continue
        for flag in fl:
            for sfx in aff.SFX.get(flag, []):
                nw = _apply_suffix(w, sfx)
                if nw and nw not in out:
                    out.add(nw)
                    if sfx.flags:
                        frontier.append((nw, sfx.flags, dep - 1))
    for flag in flags:
        for pfx in aff.PFX.get(flag, []):
            pw = _apply_prefix(stem, pfx)
            if not pw:
                continue
            out.add(pw)
            if pfx.crossproduct:
                for flag2 in flags:
                    for sfx in aff.SFX.get(flag2, []):
                        if sfx.crossproduct:
                            nw = _apply_suffix(pw, sfx)
                            if nw:
                                out.add(nw)
    return out


def expand_all(dic: Dictionary, max_len: int) -> Set[str]:
    aff = dic.aff
    forms: Set[str] = set()
    for word in dic.dic.words:
        for f in expand_stem(word.stem, word.flags, aff):
            fl = nfc_lower(f)
            if len(fl) <= max_len and WORD_RE.match(fl):
                forms.add(fl)
    return forms


def load_wordlist(path: Path, max_len: int) -> List[str]:
    words: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = nfc_lower(line)
            if w and " " not in w and len(w) <= max_len and WORD_RE.match(w):
                words.append(w)
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruir lexicon ES completo")
    parser.add_argument("--lexicon", default="data/lexicon.txt")
    parser.add_argument("--dic-base", default="data/fuentes/es")
    parser.add_argument("--backup", default="data/lexicon.prefreq.txt")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lexicon_path = Path(args.lexicon)
    dic = ensure_dictionary(Path(args.dic_base))

    print("Expandiendo diccionario hunspell...")
    hunspell_forms = expand_all(dic, args.max_len)
    print(f"  formas hunspell: {len(hunspell_forms):,}")

    current = set(load_wordlist(lexicon_path, args.max_len))
    print(f"  lexicon actual : {len(current):,}")

    # Indice de formas validas por su version sin acentos (para el dedup).
    deaccent_index: Dict[str, bool] = defaultdict(bool)
    for f in hunspell_forms:
        if any(c in f for c in ACCENTS):
            deaccent_index[strip_accents(f)] = True

    union = hunspell_forms | current
    final: Set[str] = set()
    dropped_deaccented = 0
    for w in union:
        has_accent = any(c in w for c in ACCENTS)
        if (
            not has_accent
            and w not in hunspell_forms          # no es palabra valida por si misma
            and deaccent_index.get(w, False)     # existe su variante acentuada
        ):
            dropped_deaccented += 1
            continue
        final.add(w)

    result = sorted(final)
    added = len(hunspell_forms - current)
    print(f"  + agregadas por hunspell : {added:,}")
    print(f"  - quitadas (sin tilde)   : {dropped_deaccented:,}")
    print(f"  = lexicon final          : {len(result):,}")

    if args.dry_run:
        print("(dry-run: no se escribio nada)")
        return

    backup = Path(args.backup)
    if not backup.exists():
        shutil.copy2(lexicon_path, backup)
        print(f"Backup creado: {backup}")
    else:
        print(f"Backup existente (no se sobrescribe): {backup}")

    with lexicon_path.open("w", encoding="utf-8", newline="\n") as f:
        for w in result:
            f.write(w + "\n")
    print(f"Escrito: {lexicon_path}")
    print("AHORA regenera el cache:  python tools/build_embeddings_cache.py")


if __name__ == "__main__":
    main()
