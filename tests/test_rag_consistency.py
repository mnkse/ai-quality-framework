import json
from datetime import datetime, timezone

import pytest

from ai_quality.clients.rag_client import RAGClient
from ai_quality.config.settings import PROJECT_ROOT
from ai_quality.evaluators.llm_judge import AnswerEvaluator


pytestmark = pytest.mark.live


def test_rag_return_consistency():
    client = RAGClient()
    evaluator = AnswerEvaluator()

    question = "Can I return an unused item after 90 days?"
    reference = (
        "Unused items can be returned up to and including day 30. "
        "A return after 90 days must be rejected."
    )

    runs = []

    for number in range(1, 4):
        result = client.get_answer(question)
        document_ids = sorted(doc["id"] for doc in result["documents"])

        evaluation = evaluator.evaluate(
            question=question,
            answer=result["answer"],
            reference=reference,
        )

        passed = (
            document_ids == ["return-policy"]
            and evaluation["passed"]
        )

        runs.append({
            "run": number,
            "document_ids": document_ids,
            "answer": result["answer"],
            "judge_passed": evaluation["passed"],
            "reason": evaluation["reason"],
            "passed": passed,
        })

        print(f"\nRun {number}: passed={passed}")
        print(f"Answer: {result['answer']}")
        print(f"Reason: {evaluation['reason']}")

    passed_count = sum(run["passed"] for run in runs)

    report = {
        "question": question,
        "model": client.model,
        "judge_model": evaluator.model,
        "total_runs": len(runs),
        "passed_runs": passed_count,
        "pass_rate": passed_count / len(runs),
        "runs": runs,
    }

    folder = PROJECT_ROOT / "reports"
    folder.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = folder / f"rag-consistency-{timestamp}.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nPass rate: {report['pass_rate']:.0%}")
    print(f"Report: {path}")

    assert passed_count == len(runs), (
        f"Critical policy failed in {len(runs) - passed_count} runs. "
        f"See report: {path}"
    )
