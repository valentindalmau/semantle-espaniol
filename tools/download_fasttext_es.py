from pathlib import Path

import fasttext.util


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = DATA_DIR / "cc.es.300.bin"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"Modelo ya instalado: {MODEL_PATH}")
        return

    print("Descargando modelo FastText español cc.es.300.bin...")
    fasttext.util.download_model("es", if_exists="ignore")
    downloaded = ROOT / "cc.es.300.bin"
    if not downloaded.exists():
        downloaded = Path.cwd() / "cc.es.300.bin"
    if not downloaded.exists():
        raise FileNotFoundError("No se encontro cc.es.300.bin luego de la descarga.")

    downloaded.replace(MODEL_PATH)
    print(f"Modelo instalado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
