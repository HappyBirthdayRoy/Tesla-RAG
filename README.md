# tesla-rag (T1 baseline)

Baseline RAG over **only** these two PDFs:
- `/home/node/.openclaw/media/inbound/0af5279c-11f2-4069-8708-061dfda248ae.pdf`
- `/home/node/.openclaw/media/inbound/0e52d151-9288-4c1f-893e-f4d10ec0e029.pdf`

Stack:
- Python 3.11
- ChromaDB
- `sentence-transformers/all-MiniLM-L6-v2`
- FastAPI
- pytest
- Docker

## Setup

```bash
cd /home/node/.openclaw/workspace/tesla-rag
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Ingest runbook

```bash
cd /home/node/.openclaw/workspace/tesla-rag
source .venv/bin/activate
python -m tesla_rag.cli --chroma-dir .chroma
```

Expected output: `Inserted chunks: <N>`

## Run API

```bash
cd /home/node/.openclaw/workspace/tesla-rag
source .venv/bin/activate
uvicorn tesla_rag.main:app --reload --host 0.0.0.0 --port 8000
```

## Query

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say about battery performance?"}'
```

Response includes:
- `answer` (extractive baseline from retrieved chunks)
- `citations` (`source_file`, `page`)
- `contexts` (retrieved chunks with metadata)

## Tests

```bash
cd /home/node/.openclaw/workspace/tesla-rag
source .venv/bin/activate
pytest -q
```

## Docker

Build:

```bash
docker build -t tesla-rag:latest .
```

Ingest inside container (mount inbound PDFs):

```bash
docker run --rm \
  -v /home/node/.openclaw/media/inbound:/data/inbound \
  -e TESLA_RAG_PDF_PATHS=/data/inbound/0af5279c-11f2-4069-8708-061dfda248ae.pdf,/data/inbound/0e52d151-9288-4c1f-893e-f4d10ec0e029.pdf \
  -e TESLA_RAG_ALLOW_NONDEFAULT=1 \
  tesla-rag:latest python -m tesla_rag.cli --chroma-dir /app/.chroma
```

Run API container:

```bash
docker run --rm -p 8000:8000 tesla-rag:latest
```
