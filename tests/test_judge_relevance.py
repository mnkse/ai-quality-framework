import pytest

from ai_quality.evaluators.llm_judge import AnswerEvaluator


pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def evaluator():
    return AnswerEvaluator()


@pytest.mark.parametrize(
    "answer, expected",
    [
        pytest.param(
            "Unused items can be returned up to and including day 30.",
            True,
            id="relevant-correct-answer",
        ),
        pytest.param(
            "Standard delivery takes 3 business days.",
            False,
            id="correct-fact-but-irrelevant",
        ),
        pytest.param(
            "We offer excellent customer service.",
            False,
            id="does-not-answer-question",
        ),
    ],
)
def test_judge_checks_relevance(evaluator, answer, expected):
    result = evaluator.evaluate(
        question="What is the return period for an unused item?",
        answer=answer,
        reference=(
            "Unused items can be returned up to and including day 30. "
            "Standard delivery takes 3 business days."
        ),
    )

    print(f"\nAnswer: {answer}")
    print(f"Expected decision: {expected}")
    print(f"Actual decision: {result['passed']}")
    print(f"Reason: {result['reason']}")

    assert result["passed"] is expected, result["reason"]
