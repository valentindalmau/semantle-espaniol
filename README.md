# Semantle ES

Semantle en espanol con embeddings FastText `cc.es.300.bin`, palabra diaria deterministica y ranking Top 1000 contra `data/lexicon.txt`.

## Modos

- Modo diario: la seed depende de la fecha del servidor en `APP_TIMEZONE`.
- Modo practica: permite reiniciar la partida y generar una nueva palabra.

Los intentos repetidos no se vuelven a guardar: la app resalta la fila original y muestra el numero de intento anterior.

## Correr localmente

Desde esta carpeta:

```powershell
cd "C:\Users\PC\Desktop\Semantle español\Semantle español"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools\download_fasttext_es.py
python app.py
```

Abrir:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

## Cache de embeddings

En el primer arranque la app calcula los embeddings normalizados de `data/lexicon.txt` y los guarda en:

```text
data/lexicon_embeddings.npy
data/lexicon_embeddings.npy.meta
```

En arranques posteriores carga ese cache, lo que reduce bastante el tiempo de inicio. Si cambia el lexicon, el cache se reconstruye solo.

## Variables utiles

```powershell
$env:FASTTEXT_PATH="data/cc.es.300.bin"
$env:LEXICON_EMB_CACHE_PATH="data/lexicon_embeddings.npy"
$env:TOKEN_SECRET="un-secreto-largo"
python app.py
```

Variables principales:

- `FASTTEXT_PATH`: ruta al modelo `cc.es.300.bin`.
- `TARGETS_PATH`: lista de palabras posibles del dia.
- `LEXICON_PATH`: lista usada para validar intentos y calcular el Top 1000.
- `LEXICON_EMB_CACHE_PATH`: cache local de embeddings normalizados.
- `TOP_K`: cantidad de vecinos cercanos.
- `APP_TIMEZONE`: huso horario usado para la palabra diaria.
- `TOKEN_SECRET`: firma enlaces del Top 1000. Cambiar en produccion.
- `PUBLIC_REVEAL`: permite rendirse y revelar la palabra.
- `ALLOW_RANDOM_TARGET`: permite jugar con palabra aleatoria en practica.

## GitHub

No subas `data/cc.es.300.bin` ni los caches `.npy`; estan ignorados por Git. El servidor debe descargarlos o montarlos como volumen.

## Coolify

1. Subi el repo a GitHub sin `data/cc.es.300.bin`.
2. En Coolify crea una Application desde el repo.
3. Elegi build pack `Dockerfile`.
4. Puerto interno: `8000`.
5. Usa un volumen persistente montado en `/app/data`.
6. Dentro del contenedor o desde una tarea one-off, ejecuta:

```bash
python tools/download_fasttext_es.py
```

7. Configura variables:

```text
FASTTEXT_PATH=/app/data/cc.es.300.bin
LEXICON_EMB_CACHE_PATH=/app/data/lexicon_embeddings.npy
TOKEN_SECRET=un-secreto-largo-y-unico
FLASK_DEBUG=false
```

8. Agrega un dominio en Coolify. Para dominio gratis podes usar DuckDNS o una direccion tipo `tuapp.TU-IP.sslip.io`.

El Dockerfile usa un solo worker de Gunicorn porque el modelo FastText ocupa bastante memoria.
