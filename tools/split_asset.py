"""Parte / reensambla archivos binarios grandes para poder versionarlos en git.

GitHub rechaza archivos de mas de 100 MB. El cache de embeddings (~405 MB) se
guarda en el repo como partes de <=90 MB (data/lexicon_embeddings.npy.partNN) y
el contenedor las reensambla al arrancar (ver Dockerfile).

Uso:
    python tools/split_asset.py split data/lexicon_embeddings.npy
    python tools/split_asset.py join  data/lexicon_embeddings.npy
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

CHUNK = 90 * 1024 * 1024  # 90 MB, holgado bajo el limite de 100 MB de GitHub


def split(path: str) -> None:
    for old in glob.glob(f"{path}.part*"):
        os.remove(old)
    i = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            with open(f"{path}.part{i:02d}", "wb") as out:
                out.write(data)
            i += 1
    print(f"{path} -> {i} partes de hasta {CHUNK // (1024*1024)} MB")


def join(path: str) -> None:
    parts = sorted(glob.glob(f"{path}.part*"))
    if not parts:
        raise FileNotFoundError(f"No hay partes {path}.part*")
    with open(path, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    print(f"{len(parts)} partes -> {path} ({os.path.getsize(path)} bytes)")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("split", "join"):
        print(__doc__)
        sys.exit(1)
    (split if sys.argv[1] == "split" else join)(sys.argv[2])


if __name__ == "__main__":
    main()
