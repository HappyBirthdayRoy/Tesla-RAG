# T2 Evaluation Summary

- Run ID: 20260309T073125Z-t2
- Timestamp (UTC): 2026-03-09T07:34:46Z
- Dataset path: `eval/datasets/v1_finance_qa_10.json`
- Dataset SHA256: `ab74439be2518bbac518c132d4d7d7814ef05a0caac95249abd108238f1fdd0f`
- Chroma dir: `.chroma`
- Top K: 4

## Metrics

- Exact Match: 0.0000
- Contains Gold: 0.0000
- Total Questions: 10

## Judge Metrics Scaffold

- status: `scaffold_only`
- reason: no judge model configured in V1 baseline

## Per-question

- q01: EM=NA CONTAINS=NA question="What quarter is covered by the first approved update deck PDF?" gold="2025 Q3"
- q02: EM=NA CONTAINS=NA question="What quarter is covered by the second approved update deck PDF?" gold="2025 Q4"
- q03: EM=NA CONTAINS=NA question="What is the exact title of the first approved deck?" gold="2025 Q3 Quarterly Update Deck"
- q04: EM=NA CONTAINS=NA question="What is the exact title of the second approved deck?" gold="2025 Q4 Quarterly Update Deck"
- q05: EM=NA CONTAINS=NA question="Which software created both quarterly update PDFs?" gold="Adobe InDesign 14.0 (Macintosh)"
- q06: EM=NA CONTAINS=NA question="Who is listed as the author in the Q4 quarterly update deck metadata?" gold="Travis Axelrod"
- q07: EM=NA CONTAINS=NA question="Which quarter in scope is later: Q3 or Q4 of 2025?" gold="Q4 of 2025"
- q08: EM=NA CONTAINS=NA question="How many quarterly update decks are included in this constrained corpus?" gold="2"
- q09: EM=NA CONTAINS=NA question="What is the producer string shown in the Q4 PDF metadata?" gold="macOS Version 26.2 (Build 25C56) Quartz PDFContext, AppendMode 1.1"
- q10: EM=NA CONTAINS=NA question="What created timestamp appears in the Q4 PDF metadata?" gold="D:20260128123137-08'00'"
