#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

python -m tesla_rag.eval_v1 \
  --dataset eval/datasets/v1_finance_qa_10.json \
  --chroma-dir .chroma \
  --out-dir results/t2 \
  --run-id "${RUN_ID}"
