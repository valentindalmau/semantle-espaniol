FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    LEXICON_EMB_CACHE_PATH=/app/data/lexicon_embeddings.npy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# El cache de embeddings viaja partido en data/lexicon_embeddings.npy.partNN
# (GitHub no acepta archivos >100 MB). Lo reensamblamos al arrancar, en la capa
# de escritura del contenedor, y recien ahi levantamos gunicorn.
CMD ["sh", "-c", "test -f \"$LEXICON_EMB_CACHE_PATH\" || cat data/lexicon_embeddings.npy.part* > \"$LEXICON_EMB_CACHE_PATH\"; exec gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 --timeout 180 app:app"]
