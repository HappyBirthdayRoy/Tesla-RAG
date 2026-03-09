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
- `answer` (Claude-synthesized answer constrained to retrieved chunks)
- `citations` (`source_file`, `page`)
- `contexts` (retrieved chunks with metadata)

Anthropic settings for answer synthesis:
- `ANTHROPIC_API_KEY` (required)
- `TESLA_RAG_ANTHROPIC_MODEL` (optional, default: `claude-sonnet-4-20250514`)

## Tests

```bash
cd /home/node/.openclaw/workspace/tesla-rag
source .venv/bin/activate
pytest -q
```

## V1 baseline metrics

Versioned baseline outputs are stored under `results/v1/<run-id>/`.

Current baseline artifact (Anthropic Claude reset, 2026-03-09):
- `results/v1/v1-anthropic-reset/metrics.json`
- `results/v1/v1-anthropic-reset/summary.md`

Metrics (claude-sonnet-4-20250514, top_k=4):
- Exact Match: 0.2000
- Contains Gold: 0.7000

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

## T2 evaluation harness

Dataset (10 questions, constrained to the two approved PDFs only):
- `eval/datasets/v1_finance_qa_10.json`

Judge scaffold (optional, not executed in V1):
- `eval/prompts/judge_scaffold_v1.md`

Run reproducible evaluation:

```bash
cd /home/node/.openclaw/workspace/tesla-rag
source .venv/bin/activate
python -m tesla_rag.eval_v1 \
  --dataset eval/datasets/v1_finance_qa_10.json \
  --chroma-dir .chroma \
  --out-dir results/v1 \
  --run-id v1-baseline
```

Or via wrapper:

```bash
cd /home/node/.openclaw/workspace/tesla-rag
./scripts/run_eval_v1.sh v1-baseline
```

If runtime dependencies are unavailable and you need output scaffolds only:

```bash
python -m tesla_rag.eval_v1 \
  --dataset eval/datasets/v1_finance_qa_10.json \
  --out-dir results/t2 \
  --run-id v1-scaffold \
  --scaffold-only
```

Output files per run:
- `results/v1/<run-id>/metrics.json` (machine-readable)
- `results/v1/<run-id>/summary.md` (human-readable)

Constraints:
- No external corpus/data.
- Allowed sources are explicit in dataset metadata and each item's `allowed_sources`.
