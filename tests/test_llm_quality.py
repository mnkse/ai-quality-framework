import json
from pathlib import Path

import pytest
from ai_quality.evaluators.llm_judge import AnswerEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "test_data" / "evaluation_dataset.json"

with DATA_FILE.open(encoding="utf-8-sig") as file:
    test_cases = json.load(file)


@pytest.fixture(scope="module")
def evaluator():
    return AnswerEvaluator()


@pytest.mark.parametrize(
    "case",
    test_cases,
    ids=[case["id"] for case in test_cases],
)
def test_answer_quality(case, ai_client, evaluator):
    if ai_client.mode != "live":
        pytest.skip("This test requires AI_TEST_MODE=live.")

    actual = ai_client.get_answer(case["question"])

    result = evaluator.evaluate(
        question=case["question"],
        answer=actual,
        reference=case["reference"],
    )

    print(f"\nScenario: {case['id']}")
    print(f"Chatbot answer: {actual}")
    print(f"Judge decision: {result['passed']}")
    print(f"Reason: {result['reason']}")

    assert result["passed"], (
        f"Scenario: {case['id']}\n"
        f"Answer: {actual}\n"
        f"Evaluation: {result['reason']}"
    )


import pytest

pytestmark = pytest.mark.live
