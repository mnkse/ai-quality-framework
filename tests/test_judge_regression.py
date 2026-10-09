from ai_quality.evaluators.llm_judge import AnswerEvaluator


def test_judge_accepts_clear_return_refusal():
    evaluator = AnswerEvaluator()

    result = evaluator.evaluate(
        question="Can I return an unused item after 90 days?",
        answer="No, unused items can be returned only within 30 days.",
        reference="Unused items can be returned within 30 days.",
    )

    print(f"\nJudge decision: {result['passed']}")
    print(f"Reason: {result['reason']}")

    assert result["passed"], result["reason"]


import pytest

pytestmark = pytest.mark.live
