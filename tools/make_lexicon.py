# tools/make_lexicon.py
from __future__ import annotations
from typing import Optional, Iterable, Set, List, Dict
from pathlib import Path
import argparse, time, gzip, unicodedata, re, json
import requests
from tqdm import tqdm
from wordfreq import top_n_list, zipf_frequency
from unidecode import unidecode
import spacy

# ----------------- Config por defecto -----------------
WIKI_TITLES_URL = "https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-all-titles-in-ns0.gz"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
# Clases Wikidata a traer (empresa, organización, marca, producto)
WIKIDATA_QIDS = {
    "empresa": "Q4830453",
    "organizacion": "Q43229",
    "marca": "Q431289",
    "producto": "Q2424752",
}
MIN_ZIPF = 2.5   # umbral de frecuencia para aceptar palabras fuera de listas
MAX_TITLE_LEN = 40

RE_TOKEN = re.compile(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9 .&'’\-]{2,40}$")
RE_SIGLA_PURA = re.compile(r"^[A-Z]{2,6}$")
RE_SIGLA_PUNTOS = re.compile(r"^(?:[A-Z]\.){2,6}$")

# ----------------- Utilidades -----------------
def nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip().lower()

def clean_entry(s: str) -> Optional[str]:
    s = nfc_lower(s)
    s = s.replace("\u200b", "")
    if not s or s.startswith("#"):
        return None
    if RE_TOKEN.match(s) is None:
        return None
    return s

def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def iter_lines_txt_or_gz(p: Path) -> Iterable[str]:
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
    else:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line

# ----------------- wordfreq -----------------
def load_wordfreq_top(n: int) -> Set[str]:
    words = set()
    for w in top_n_list("es", n):
        cw = clean_entry(w)
        if cw:
            words.add(cw)
    return words

# ----------------- Wikipedia títulos -----------------
def load_eswiki_titles(cache_dir: Path) -> Set[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / "eswiki-latest-all-titles-in-ns0.gz"
    if not gz_path.exists():
        print("Descargando títulos de Wikipedia ES…")
        download_file(WIKI_TITLES_URL, gz_path)
    print("Procesando títulos de Wikipedia ES…")
    titles: Set[str] = set()
    for line in tqdm(iter_lines_txt_or_gz(gz_path), desc="wiki titles"):
        title = line.strip()
        if not title:
            continue
        # descartamos espacios de nombres raros (p.ej. "Categoría:", aunque en ns0 casi no hay)
        # normalizamos y filtramos largo
        title = title.replace("_", " ")
        if len(title) > MAX_TITLE_LEN:
            continue
        ce = clean_entry(title)
        if ce:
            titles.add(ce)
    return titles

def extract_siglas_from_titles(titles: Iterable[str]) -> Set[str]:
    siglas: Set[str] = set()
    for t in titles:
        raw = t.strip()
        upper = raw.upper()
        if RE_SIGLA_PURA.match(upper) or RE_SIGLA_PUNTOS.match(upper):
            siglas.add(upper)
    return siglas

# ----------------- Wikidata (SPARQL por páginas) -----------------
def fetch_wikidata_labels_for(qid: str, lang: str = "es",
                              page_size: int = 10000,
                              max_pages: int = 20,
                              sleep_s: float = 1.0) -> Set[str]:
    """
    Trae labels en español (con fallback) para entidades instancia-de (P31) qid (o subclases).
    Evita KeyError cuando no hay label.
    """
    all_labels: Set[str] = set()
    # Idiomas: auto (según entidad), español y como respaldo inglés
    langs = f"[AUTO_LANGUAGE],{lang},en"

    for page in range(max_pages):
        offset = page * page_size
        query = f"""
        SELECT ?e ?eLabel WHERE {{
          ?e wdt:P31/wdt:P279* wd:{qid} .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{langs}" . }}
        }} LIMIT {page_size} OFFSET {offset}
        """

        # reintentos simples por rate limit o errores transitorios
        for attempt in range(4):
            try:
                r = requests.get(
                    WIKIDATA_ENDPOINT,
                    params={"query": query, "format": "json"},
                    headers={"User-Agent": "semantle-es-lexicon-builder/1.0"},
                    timeout=60
                )
                # 429/5xx → reintentar
                if r.status_code in (429, 502, 503, 504):
                    wait = sleep_s * (attempt + 1)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except requests.RequestException:
                wait = sleep_s * (attempt + 1)
                time.sleep(wait)
        else:
            # agotados los reintentos
            break

        data = r.json()
        rows = data.get("results", {}).get("bindings", [])
        if not rows:
            break

        for row in rows:
            # Puede no venir eLabel → saltar fila
            label = row.get("eLabel", {}).get("value")
            if not label:
                continue
            ce = clean_entry(label)
            if ce:
                all_labels.add(ce)

        time.sleep(sleep_s)  # aflojamos para no saturar el endpoint

    return all_labels


def fetch_wikidata_block(lang: str = "es") -> Set[str]:
    labels: Set[str] = set()
    print("Consultando Wikidata (empresa, organización, marca, producto)…")
    for name, q in WIKIDATA_QIDS.items():
        print(f"  • {name} ({q})")
        block = fetch_wikidata_labels_for(q, lang=lang)
        print(f"    +{len(block):,} labels")
        labels |= block
    return labels


# ----------------- Heurística de aceptación por frecuencia -----------------
def accept_by_zipf(w: str, lang: str = "es", min_zipf: float = MIN_ZIPF) -> bool:
    if zipf_frequency(w, lang) >= min_zipf:  # con tildes
        return True
    if zipf_frequency(unidecode(w), lang) >= min_zipf:  # sin tildes
        return True
    return False

# ----------------- Targets POS -----------------
def make_targets_from_lexicon(lexicon: Iterable[str], out_path: Path, allow_propn: bool = True) -> int:
    print("Filtrando targets (ADJ/NOUN/PROPN) con spaCy…")
    nlp = spacy.load("es_core_news_sm")
    pos_keep = {"ADJ", "NOUN"} | ({"PROPN"} if allow_propn else set())

    targets: Set[str] = set()
    for w in tqdm(lexicon, total=len(lexicon), desc="spaCy POS"):
        if " " in w:
            continue
        doc = nlp(w)
        if len(doc) == 1 and doc[0].pos_ in pos_keep:
            targets.add(w)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for w in sorted(targets):
            f.write(w + "\n")
    return len(targets)

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser(description="Builder de lexicón gigante ES")
    parser.add_argument("--cache", type=str, default="data/fuentes", help="Carpeta de cache/descargas")
    parser.add_argument("--out", type=str, default="data/lexicon_es.txt", help="Salida lexicón")
    parser.add_argument("--targets", type=str, default="data/targets_es.txt", help="Salida targets")
    parser.add_argument("--top-wordfreq", type=int, default=500000, help="N palabras top de wordfreq (es)")
    parser.add_argument("--skip-wiki", action="store_true", help="No traer títulos de Wikipedia")
    parser.add_argument("--skip-wikidata", action="store_true", help="No consultar Wikidata")
    parser.add_argument("--allow-propn", action="store_true", help="Incluir nombres propios (PROPN) en targets. No recomendado: mete vladimir, putin, etc.")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    out_lex = Path(args.out)
    out_targets = Path(args.targets)

    lex: Set[str] = set()

    # 1) wordfreq
    print(f"Cargando wordfreq top {args.top_wordfreq}…")
    lex |= load_wordfreq_top(args.top_wordfreq)

    # 2) Wikipedia titles
    siglas: Set[str] = set()
    if not args.skip_wiki:
        titles = load_eswiki_titles(cache_dir)
        lex |= titles
        siglas |= extract_siglas_from_titles(titles)
        # también aceptamos títulos muy frecuentes por si quedaron fuera del regex
        print("Validando títulos por frecuencia…")
        extra_freq = {t for t in titles if accept_by_zipf(t)}
        lex |= extra_freq

    # 3) Wikidata labels ES
    if not args.skip_wikidata:
        labels = fetch_wikidata_block(lang="es")
        lex |= labels

    # 4) Agregar siglas detectadas
    if siglas:
        print(f"Agregando {len(siglas)} siglas detectadas…")
        lex |= {s.lower() for s in siglas}  # normalizamos a lower para el set

    # 5) Dump lexicon
    out_lex.parent.mkdir(parents=True, exist_ok=True)
    with out_lex.open("w", encoding="utf-8") as f:
        for w in sorted(lex):
            f.write(w + "\n")
    print(f"OK → {out_lex} ({len(lex):,} entradas)")

    # 6) Targets POS
    n_targets = make_targets_from_lexicon(list(lex), out_targets, allow_propn=args.allow_propn)
    print(f"OK → {out_targets} ({n_targets:,} targets)")
    print("IMPORTANTE: corre 'python tools/curate_targets.py' para sacar nombres "
          "propios, siglas y extranjerismos que el POS de spaCy deja pasar.")

if __name__ == "__main__":
    main()
