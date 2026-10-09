import pytest
from evaluator import AnswerEvaluator


@pytest.fixture(scope="module")
def evaluator():
    return AnswerEvaluator()


@pytest.mark.parametrize(
    "answer, expected",
    [
        pytest.param(
            "No, you cannot return an unused item after 90 days. "
            "Returns are only accepted within 30 days.",
            True,
            id="correct-rejection",
        ),
        pytest.param(
            "Yes, you can return an unused item after 90 days.",
            False,
            id="incorrect-permission",
        ),
        pytest.param(
            "No, returns are only accepted within 60 days.",
            False,
            id="incorrect-policy-limit",
        ),
    ],
)
def test_judge_understands_policy(evaluator, answer, expected):
    result = evaluator.evaluate(
        question="Can I return an unused item after 90 days?",
        answer=answer,
        reference=(
            "Unused items can be returned within 30 days. "
            "Returns after 30 days are not accepted."
        ),
    )

    print(f"\nAnswer: {answer}")
    print(f"Decision: {result['passed']}")
    print(f"Reason: {result['reason']}")

    assert result["passed"] is expected, result["reason"]
