# V1 Evaluation Summary

- Run ID: v4-hybrid-bm25w135-20260309T1636Z
- Timestamp (UTC): 2026-03-09T16:38:23Z
- Dataset path: `eval/datasets/v1_finance_qa_10.json`
- Dataset SHA256: `d6a1db931165fa75a826ab85904c16d096d61369f37b04610dbc2bbe8077d7b3`
- Chroma dir: `.chroma_v2_20260309T131817Z`
- Top K: 4

## Metrics

- Exact Match: 0.1000
- Contains Gold: 0.4000
- Total Questions: 10

## Judge Metrics Scaffold

- status: `not_executed`
- reason: no judge model configured in V1 baseline

## Per-question

- q01: EM=0 CONTAINS=0 question="In the Q3 2025 financial summary, what is total revenues (in millions of USD)?" gold="28,095"
- q02: EM=0 CONTAINS=0 question="In the Q3 2025 financial summary, what is energy generation and storage revenue (in millions of USD)?" gold="3,415"
- q03: EM=0 CONTAINS=1 question="What is total vehicle deliveries in Q3 2025?" gold="497,099"
- q04: EM=0 CONTAINS=0 question="On the Q3 2025 balance sheet, what is cash, cash equivalents and investments (in millions of USD)?" gold="41,647"
- q05: EM=0 CONTAINS=0 question="In Q3 2025 cash flows from operating activities, what is depreciation, amortization and impairment (in millions of USD)?" gold="1,625"
- q06: EM=0 CONTAINS=0 question="In the Q4 2025 quarterly financial summary, what is total revenues (in millions of USD)?" gold="24,901"
- q07: EM=0 CONTAINS=1 question="In FY 2025 financial summary, what is total revenues (in millions of USD)?" gold="94,827"
- q08: EM=0 CONTAINS=1 question="In FY 2025 financial summary, what is energy generation and storage revenue (in millions of USD)?" gold="12,771"
- q09: EM=1 CONTAINS=1 question="In Q4 2025 financial statements, what is net income attributable to common stockholders (GAAP) (in millions of USD)?" gold="840"
- q10: EM=0 CONTAINS=0 question="On the 31-Dec-25 balance sheet, what is cash, cash equivalents and investments (in millions of USD)?" gold="44,059"
