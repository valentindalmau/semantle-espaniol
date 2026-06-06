"""Cura data/targets_es.txt para que solo queden palabras adivinables.

Elimina nombres propios (vladimir, putin), siglas (adn, dni), fragmentos (abr,
aca), extranjerismos no aceptados (net, online) y formas mal acentuadas
(administracion). Para eso exige que cada target:

  1. este en el lexicon (asi el embedding ya existe; no hay que reconstruir cache),
  2. sea reconocido por el diccionario hunspell del espanol (spylls), que aplica
     reglas de afijos y por ende acepta formas flexionadas (casas, montañas,
     corriendo) y rechaza nombres propios / siglas / extranjerismos,
  3. tenga una frecuencia minima (wordfreq) para asegurar que sea "conocido".

Los prestamos aceptados por la RAE (rock, jazz, software, web) quedan a proposito:
son palabras reales del espanol, no imposibles.

Uso:
    python tools/curate_targets.py                 # cura in-place con backup
    python tools/curate_targets.py --min-zipf 3.3  # mas estricto (menos targets)
    python tools/curate_targets.py --dry-run       # solo reporta, no escribe
"""
from __future__ import annotations

import argparse
import shutil
import unicodedata
from pathlib import Path
from typing import List, Set

import requests
from spylls.hunspell import Dictionary
from wordfreq import zipf_frequency

DIC_BASE_URL = "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/es/index"


def nfc_lower(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip().lower())


def load_words(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [nfc_lower(line) for line in f if line.strip()]


def ensure_dictionary(base: Path) -> Dictionary:
    """Descarga es.dic/es.aff si faltan y devuelve el diccionario hunspell."""
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("dic", "aff"):
        target = base.with_suffix(f".{ext}")
        if not target.exists():
            print(f"Descargando diccionario es ({ext})...")
            resp = requests.get(f"{DIC_BASE_URL}.{ext}", timeout=60)
            resp.raise_for_status()
            target.write_text(resp.text, encoding="utf-8")
    return Dictionary.from_files(str(base))


def main() -> None:
    parser = argparse.ArgumentParser(description="Curar targets de Semantle ES")
    parser.add_argument("--targets", default="data/targets_es.txt")
    parser.add_argument("--lexicon", default="data/lexicon.txt")
    parser.add_argument("--dic-base", default="data/fuentes/es", help="Base de es.dic/es.aff")
    parser.add_argument("--backup", default="data/targets_es.bak.txt")
    parser.add_argument("--min-zipf", type=float, default=2.5, help="Frecuencia minima (wordfreq es)")
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets_path = Path(args.targets)
    lexicon_path = Path(args.lexicon)

    targets = load_words(targets_path)
    lexicon: Set[str] = set(load_words(lexicon_path))
    dic = ensure_dictionary(Path(args.dic_base))

    kept: List[str] = []
    dropped_examples: List[str] = []
    for word in targets:
        ok = (
            len(word) >= args.min_len
            and word in lexicon
            and bool(dic.lookup(word))
            and zipf_frequency(word, "es") >= args.min_zipf
        )
        if ok:
            kept.append(word)
        elif len(dropped_examples) < 40:
            dropped_examples.append(word)

    kept = sorted(set(kept))

    print(f"Targets originales : {len(targets):,}")
    print(f"Targets curados    : {len(kept):,}")
    print(f"Eliminados         : {len(targets) - len(kept):,}")
    print(f"Ejemplos eliminados: {dropped_examples}")

    if args.dry_run:
        print("(dry-run: no se escribio nada)")
        return

    backup = Path(args.backup)
    if not backup.exists():
        shutil.copy2(targets_path, backup)
        print(f"Backup creado: {backup}")
    else:
        print(f"Backup existente (no se sobrescribe): {backup}")

    with targets_path.open("w", encoding="utf-8", newline="\n") as f:
        for word in kept:
            f.write(word + "\n")
    print(f"Escrito: {targets_path}")


if __name__ == "__main__":
    main()
