import os
import json
from datetime import datetime, timezone

import pytest
from dotenv import dotenv_values

from ai_quality.clients.rag_client import RAGClient
from ai_quality.config.settings import ENV_FILE, PROJECT_ROOT


pytestmark = pytest.mark.live


def test_rag_deepeval_quality(monkeypatch):
    settings = dotenv_values(ENV_FILE, encoding="utf-8-sig")
    api_key = settings.get("OPENAI_API_KEY")
    judge_model = (os.getenv("DEEPEVAL_JUDGE_MODEL") or settings.get("DEEPEVAL_JUDGE_MODEL") or settings.get("JUDGE_MODEL"))

    if not api_key or not judge_model:
        raise ValueError("OPENAI_API_KEY or JUDGE_MODEL is missing.")

    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    from ai_quality.evaluators.faithfulness_template import PolicyFaithfulnessTemplate

    question = "Can I return an unused item after 90 days?"
    result = RAGClient().get_answer(question)

    document_ids = sorted(doc["id"] for doc in result["documents"])
    assert document_ids == ["return-policy"]

    case = LLMTestCase(
        input=question,
        actual_output=result["answer"],
        retrieval_context=[
            doc["content"] for doc in result["documents"]
        ],
    )

    options = {
        "model": judge_model,
        "threshold": 0.7,
        "include_reason": True,
        "async_mode": False,
        "eval_mode": "llm",
    }

    metrics = {
        "answer_relevancy": AnswerRelevancyMetric(**options),
        "faithfulness": FaithfulnessMetric(**options, evaluation_template=PolicyFaithfulnessTemplate),
    }

    scores = {}

    for name, metric in metrics.items():
        metric.measure(case)
        scores[name] = {
            "score": metric.score,
            "threshold": metric.threshold,
            "reason": metric.reason,
        }
        print(f"\n{name}: {metric.score}")
        print(f"Reason: {metric.reason}")

    report = {
        "question": question,
        "answer": result["answer"],
        "documents": result["documents"],
        "telemetry": result["telemetry"],
        "judge_model": judge_model,
        "eval_mode": "llm",
        "metrics": scores,
    }

    folder = PROJECT_ROOT / "reports"
    folder.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = folder / f"rag-deepeval-{timestamp}.json"

    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nRAG answer: {result['answer']}")
    print(f"Report: {path}")

    for name, measurement in scores.items():
        score = measurement["score"]
        assert score is not None, f"{name}: no score returned"
        assert 0 <= score <= 1, f"{name}: invalid score"
        assert score >= measurement["threshold"], (
            f"{name}: {measurement['reason']}"
        )
