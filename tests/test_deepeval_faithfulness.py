import os
import pytest
from dotenv import dotenv_values

from ai_quality.config.settings import ENV_FILE


pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "answer, expected_pass",
    [
        pytest.param(
            "No, you cannot return an unused item after 90 days. "
            "Returns are only accepted up to and including day 30 "
            "after purchase.",
            True,
            id="correct-90-day-rejection",
        ),
        pytest.param(
            "Unused items can be returned through day 30.",
            True,
            id="matches-source",
        ),
        pytest.param(
            "Unused items can be returned through day 60.",
            False,
            id="contradicts-source",
        ),
    ],
)
def test_deepeval_faithfulness(monkeypatch, answer, expected_pass):
    settings = dotenv_values(ENV_FILE, encoding="utf-8-sig")
    api_key = settings.get("OPENAI_API_KEY")
    judge_model = (os.getenv("DEEPEVAL_JUDGE_MODEL") or settings.get("DEEPEVAL_JUDGE_MODEL") or settings.get("JUDGE_MODEL"))

    if not api_key or not judge_model:
        raise ValueError("OPENAI_API_KEY or JUDGE_MODEL is missing.")

    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    from deepeval.metrics import FaithfulnessMetric
    from ai_quality.evaluators.faithfulness_template import PolicyFaithfulnessTemplate
    from deepeval.test_case import LLMTestCase

    case = LLMTestCase(
        input="What is the return period for an unused item?",
        actual_output=answer,
        retrieval_context=[
            "Unused items can be returned up to and including day 30. "
            "Returns on day 31 or later are not accepted."
        ],
    )

    metric = FaithfulnessMetric(
        model=judge_model,
        threshold=0.7,
        include_reason=True,
        async_mode=False,
        verbose_mode=True,
        eval_mode="llm",
        evaluation_template=PolicyFaithfulnessTemplate,
    )

    metric.measure(case)

    for field in ("truths", "claims", "verdicts"):
        print(f"\n{field.upper()}:")
        print(getattr(metric, field, "Not available"))

    print(f"\nAnswer: {answer}")
    print(f"Faithfulness score: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert metric.score is not None
    assert 0 <= metric.score <= 1

    accepted = metric.score >= metric.threshold
    assert accepted is expected_pass, metric.reason
