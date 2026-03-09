from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tesla_rag.config import DEFAULT_CHROMA_DIR, DEFAULT_TOP_K


@dataclass(slots=True)
class EvalItem:
    id: str
    question: str
    answer: str
    allowed_sources: list[str]


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    collapsed = re.sub(r"\s+", " ", lowered)
    stripped = re.sub(r"[^a-z0-9\s\-\.:']", "", collapsed)
    return stripped


def _exact_match(prediction: str, gold: str) -> bool:
    return _normalize(prediction) == _normalize(gold)


def _contains_gold(prediction: str, gold: str) -> bool:
    return _normalize(gold) in _normalize(prediction)


def _dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_dataset(path: Path) -> tuple[dict, list[EvalItem]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    items_raw = payload.get("items", [])
    items: list[EvalItem] = []
    for row in items_raw:
        items.append(
            EvalItem(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                allowed_sources=row.get("allowed_sources", []),
            )
        )
    if len(items) != 10:
        raise ValueError(f"Expected exactly 10 eval questions, found {len(items)}")
    return meta, items


def _now_utc_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _build_result_dir(base: Path, run_id: str | None) -> Path:
    out_dir = base / (run_id or _now_utc_compact())
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_summary_md(path: Path, metrics: dict, per_question: list[dict], run_meta: dict, judge_status: str) -> None:
    lines: list[str] = []
    lines.append("# T2 Evaluation Summary")
    lines.append("")
    lines.append(f"- Run ID: {run_meta['run_id']}")
    lines.append(f"- Timestamp (UTC): {run_meta['timestamp_utc']}")
    lines.append(f"- Dataset path: `{run_meta['dataset_path']}`")
    lines.append(f"- Dataset SHA256: `{run_meta['dataset_sha256']}`")
    lines.append(f"- Chroma dir: `{run_meta['chroma_dir']}`")
    lines.append(f"- Top K: {run_meta['top_k']}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- Exact Match: {metrics['exact_match']:.4f}")
    lines.append(f"- Contains Gold: {metrics['contains_gold']:.4f}")
    lines.append(f"- Total Questions: {metrics['total_questions']}")
    lines.append("")
    lines.append("## Judge Metrics Scaffold")
    lines.append("")
    lines.append(f"- status: `{judge_status}`")
    lines.append("- reason: no judge model configured in V1 baseline")
    lines.append("")
    lines.append("## Per-question")
    lines.append("")
    for row in per_question:
        em = "NA" if row["exact_match"] is None else str(int(row["exact_match"]))
        contains = "NA" if row["contains_gold"] is None else str(int(row["contains_gold"]))
        lines.append(
            f"- {row['id']}: EM={em} CONTAINS={contains} "
            f"question=\"{row['question']}\" gold=\"{row['gold_answer']}\""
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eval(
    dataset_path: Path,
    chroma_dir: Path,
    out_base_dir: Path,
    run_id: str | None,
    top_k: int,
    scaffold_only: bool,
) -> dict:
    dataset_meta, items = _load_dataset(dataset_path)
    if scaffold_only:
        per_question = [
            {
                "id": item.id,
                "question": item.question,
                "gold_answer": item.answer,
                "prediction": None,
                "citations": [],
                "citation_sources": [],
                "allowed_sources": item.allowed_sources,
                "exact_match": None,
                "contains_gold": None,
                "source_constraint_ok": None,
            }
            for item in items
        ]
        metrics = {
            "exact_match": 0.0,
            "contains_gold": 0.0,
            "total_questions": len(items),
            "exact_match_hits": 0,
            "contains_gold_hits": 0,
        }
    else:
        from tesla_rag.service import RagService

        service = RagService(persist_dir=str(chroma_dir))
        if service.store.count() == 0:
            raise RuntimeError(
                f"Chroma index is empty at {chroma_dir}. Run ingestion first with the two allowed PDFs."
            )

        exact_match_hits = 0
        contains_hits = 0
        per_question = []

        for item in items:
            result = service.answer(question=item.question, top_k=top_k)
            prediction = result["answer"]
            citations = result["citations"]
            citation_sources = sorted({c["source_file"] for c in citations})

            em = _exact_match(prediction, item.answer)
            contains = _contains_gold(prediction, item.answer)
            exact_match_hits += int(em)
            contains_hits += int(contains)

            per_question.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "gold_answer": item.answer,
                    "prediction": prediction,
                    "citations": citations,
                    "citation_sources": citation_sources,
                    "allowed_sources": item.allowed_sources,
                    "exact_match": em,
                    "contains_gold": contains,
                    "source_constraint_ok": all(src in item.allowed_sources for src in citation_sources),
                }
            )

        total = len(items)
        metrics = {
            "exact_match": exact_match_hits / total if total else 0.0,
            "contains_gold": contains_hits / total if total else 0.0,
            "total_questions": total,
            "exact_match_hits": exact_match_hits,
            "contains_gold_hits": contains_hits,
        }

    run_key = run_id or _now_utc_compact()
    out_dir = _build_result_dir(out_base_dir, run_key)
    timestamp_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    output = {
        "version": "t2-v1-eval",
        "run": {
            "run_id": run_key,
            "timestamp_utc": timestamp_utc,
            "dataset_path": str(dataset_path),
            "dataset_sha256": _dataset_sha256(dataset_path),
            "chroma_dir": str(chroma_dir),
            "top_k": top_k,
        },
        "constraints": {
            "dataset_meta": dataset_meta,
            "external_corpus_used": False,
        },
        "metrics": metrics,
        "judge_metrics": {
            "status": "not_executed" if not scaffold_only else "scaffold_only",
            "reason": "no judge model configured in V1 baseline",
            "rubric_path": "eval/prompts/judge_scaffold_v1.md",
        },
        "per_question": per_question,
    }

    metrics_json = out_dir / "metrics.json"
    summary_md = out_dir / "summary.md"
    metrics_json.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_summary_md(
        summary_md,
        metrics=metrics,
        per_question=per_question,
        run_meta=output["run"],
        judge_status=output["judge_metrics"]["status"],
    )
    return {"out_dir": str(out_dir), "metrics_json": str(metrics_json), "summary_md": str(summary_md)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V1 baseline evaluation against a fixed 10-question dataset.")
    parser.add_argument("--dataset", default="eval/datasets/v1_finance_qa_10.json")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--out-dir", default="results/t2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--scaffold-only", action="store_true")
    args = parser.parse_args()

    result = run_eval(
        dataset_path=Path(args.dataset),
        chroma_dir=Path(args.chroma_dir),
        out_base_dir=Path(args.out_dir),
        run_id=args.run_id,
        top_k=args.top_k,
        scaffold_only=args.scaffold_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
