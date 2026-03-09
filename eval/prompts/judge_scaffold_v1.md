# Judge Scaffold (V1)

Purpose: optional rubric for a future LLM judge pass over prediction quality.

Current baseline behavior:
- V1 does not execute LLM judge scoring.
- Evaluator writes `judge_metrics.status = not_executed`.

Intended rubric dimensions (0-5 each):
- Faithfulness to retrieved contexts
- Correctness against gold answer
- Citation usefulness (source + page quality)
- Conciseness

Expected input:
- question
- gold_answer
- model_prediction
- retrieved_contexts
- citations

Expected output JSON shape:
{
  "faithfulness": 0,
  "correctness": 0,
  "citation_quality": 0,
  "conciseness": 0,
  "notes": ""
}
