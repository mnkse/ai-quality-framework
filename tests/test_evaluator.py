import pytest
from evaluator import AnswerEvaluator


@pytest.fixture(scope="module")
def evaluator():
    return AnswerEvaluator()


@pytest.mark.parametrize(
    "answer, expected_pass",
    [
        pytest.param(
            "No. The return window is thirty days, so 90 days is too late.",
            True,
            id="correct-paraphrase",
        ),
        pytest.param(
            "Yes, you can return it after 90 days despite the 30 days rule.",
            False,
            id="incorrect-answer",
        ),
    ],
)
def test_evaluator_decision(evaluator, answer, expected_pass):
    result = evaluator.evaluate(
        question="Can I return an unused item after 90 days?",
        answer=answer,
        reference="Unused items can be returned within 30 days.",
    )

    print(f"\nAnswer: {answer}")
    print(f"Judge decision: {result['passed']}")
    print(f"Reason: {result['reason']}")

    assert result["passed"] is expected_pass, result["reason"]


import pytest

pytestmark = pytest.mark.live
