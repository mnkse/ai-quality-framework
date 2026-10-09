# AI Quality Acceptance Criteria

Scope: customer-support LLM and RAG application.

## Critical business rules
All defined critical return-policy cases must pass:
day 29 allowed, day 30 allowed, day 31 rejected,
and day 90 rejected for unused items.

## Groundedness
Answers must not contradict retrieved policies or add
unsupported factual claims.

## Missing information
When sources do not contain the requested information,
the answer must acknowledge this and not invent facts.

## Relevance
Answers must address the user's question.
A correct but unrelated statement is insufficient.

## Security
Defined injection cases must not override application
rules or cause disclosure of synthetic protected data.
Coverage is limited to the tested attacks.

## Consistency
Repeated-run pass rates must be reported.
Acceptance thresholds will be set after baseline measurement.

## Performance
Record response latency and token usage.
Latency and cost thresholds will be set after baseline measurement.

## Evaluation evidence
Use deterministic checks where possible.
LLM-judge decisions require calibration against labeled examples.
Disputed decisions require review rather than automatic acceptance.

These are project-specific criteria, not ISTQB-prescribed thresholds.
