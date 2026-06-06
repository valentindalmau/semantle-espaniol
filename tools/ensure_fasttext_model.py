from pathlib import Path
import os
import shutil

import fasttext.util


def main() -> None:
    model_path = Path(os.getenv("FASTTEXT_PATH", "data/cc.es.300.bin"))
    if model_path.exists():
        print(f"Modelo FastText encontrado: {model_path}", flush=True)
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"No existe {model_path}. Descargando cc.es.300.bin...", flush=True)
    fasttext.util.download_model("es", if_exists="ignore")

    downloaded = Path("cc.es.300.bin")
    if not downloaded.exists():
        raise FileNotFoundError("La descarga termino pero no se encontro cc.es.300.bin")

    shutil.move(str(downloaded), str(model_path))
    print(f"Modelo FastText instalado en: {model_path}", flush=True)


if __name__ == "__main__":
    main()
