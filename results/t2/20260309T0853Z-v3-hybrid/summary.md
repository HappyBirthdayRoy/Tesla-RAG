# T2 Evaluation Summary

- Run ID: 20260309T0853Z-v3-hybrid
- Timestamp (UTC): 2026-03-09T08:58:04Z
- Dataset path: `eval/datasets/v1_finance_qa_10.json`
- Dataset SHA256: `ab74439be2518bbac518c132d4d7d7814ef05a0caac95249abd108238f1fdd0f`
- Chroma dir: `.chroma_v2`
- Top K: 4

## Metrics

- Exact Match: 0.0000
- Contains Gold: 0.1000
- Total Questions: 10

## Judge Metrics Scaffold

- status: `not_executed`
- reason: no judge model configured in V1 baseline

## Per-question

- q01: EM=0 CONTAINS=0 question="What quarter is covered by the first approved update deck PDF?" gold="2025 Q3"
- q02: EM=0 CONTAINS=0 question="What quarter is covered by the second approved update deck PDF?" gold="2025 Q4"
- q03: EM=0 CONTAINS=0 question="What is the exact title of the first approved deck?" gold="2025 Q3 Quarterly Update Deck"
- q04: EM=0 CONTAINS=0 question="What is the exact title of the second approved deck?" gold="2025 Q4 Quarterly Update Deck"
- q05: EM=0 CONTAINS=0 question="Which software created both quarterly update PDFs?" gold="Adobe InDesign 14.0 (Macintosh)"
- q06: EM=0 CONTAINS=0 question="Who is listed as the author in the Q4 quarterly update deck metadata?" gold="Travis Axelrod"
- q07: EM=0 CONTAINS=0 question="Which quarter in scope is later: Q3 or Q4 of 2025?" gold="Q4 of 2025"
- q08: EM=0 CONTAINS=1 question="How many quarterly update decks are included in this constrained corpus?" gold="2"
- q09: EM=0 CONTAINS=0 question="What is the producer string shown in the Q4 PDF metadata?" gold="macOS Version 26.2 (Build 25C56) Quartz PDFContext, AppendMode 1.1"
- q10: EM=0 CONTAINS=0 question="What created timestamp appears in the Q4 PDF metadata?" gold="D:20260128123137-08'00'"

## V2 vs V3 Comparison

- V2 Run: 20260309T0825Z-v2-chunk
- V3 Run: 20260309T0853Z-v3-hybrid
- Exact Match: V2=0.0000 | V3=0.0000
- Contains Gold: V2=0.1000 | V3=0.1000
