import os
import pytest
from dotenv import dotenv_values

from ai_quality.config.settings import ENV_FILE


pytestmark = pytest.mark.live


def test_deepeval_answer_relevancy(monkeypatch):
    settings = dotenv_values(ENV_FILE, encoding="utf-8-sig")
    api_key = settings.get("OPENAI_API_KEY")
    judge_model = (os.getenv("DEEPEVAL_JUDGE_MODEL") or settings.get("DEEPEVAL_JUDGE_MODEL") or settings.get("JUDGE_MODEL"))

    if not api_key or not judge_model:
        raise ValueError("OPENAI_API_KEY or JUDGE_MODEL is missing.")

    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    # Import inside the live test so offline CI does not require DeepEval.
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    case = LLMTestCase(
        input="What is the return period for an unused item?",
        actual_output="Unused items can be returned through day 30.",
    )

    metric = AnswerRelevancyMetric(
        model=judge_model,
        threshold=0.7,
        include_reason=True,
        async_mode=False,
        eval_mode="llm",
    )

    metric.measure(case)

    print(f"\nJudge model: {judge_model}")
    print(f"Answer relevancy score: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert metric.score is not None
    assert 0 <= metric.score <= 1
    assert metric.score >= metric.threshold, metric.reason


def test_deepeval_rejects_irrelevant_answer(monkeypatch):
    settings = dotenv_values(ENV_FILE, encoding="utf-8-sig")
    api_key = settings.get("OPENAI_API_KEY")
    judge_model = (os.getenv("DEEPEVAL_JUDGE_MODEL") or settings.get("DEEPEVAL_JUDGE_MODEL") or settings.get("JUDGE_MODEL"))

    if not api_key or not judge_model:
        raise ValueError("OPENAI_API_KEY or JUDGE_MODEL is missing.")

    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    case = LLMTestCase(
        input="What is the return period for an unused item?",
        actual_output="Standard delivery takes 3 business days.",
    )

    metric = AnswerRelevancyMetric(
        model=judge_model,
        threshold=0.7,
        include_reason=True,
        async_mode=False,
        eval_mode="llm",
    )

    metric.measure(case)

    print(f"\nAnswer relevancy score: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert metric.score is not None
    assert 0 <= metric.score <= 1
    assert metric.score < metric.threshold, (
        f"Irrelevant answer unexpectedly accepted: {metric.reason}"
    )
