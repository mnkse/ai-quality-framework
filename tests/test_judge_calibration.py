import json
from pathlib import Path

import pytest

from ai_quality.config.settings import PROJECT_ROOT
from ai_quality.evaluators.classification_metrics import calculate_metrics
from ai_quality.evaluators.llm_judge import AnswerEvaluator


pytestmark = pytest.mark.live


def test_judge_calibration():
    path = PROJECT_ROOT / "test_data" / "judge_calibration.json"
    cases = json.loads(path.read_text(encoding="utf-8-sig"))

    evaluator = AnswerEvaluator()
    labels = []
    predictions = []
    mismatches = []

    for case in cases:
        result = evaluator.evaluate(
            question="What is the return period for an unused item?",
            answer=case["answer"],
            reference=(
                "Unused items can be returned up to and including day 30. "
                "Returns on day 31 or later are not accepted. "
                "Standard delivery takes 3 business days."
            ),
        )

        labels.append(case["expected"])
        predictions.append(result["passed"])

        print(
            f"\n{case['id']}: expected={case['expected']}, "
            f"actual={result['passed']}"
        )
        print(f"Reason: {result['reason']}")

        if result["passed"] is not case["expected"]:
            mismatches.append(case["id"])

    metrics = calculate_metrics(labels, predictions)
    print("\nJudge calibration metrics:")
    print(json.dumps(metrics, indent=2))

    assert not mismatches, f"Misclassified cases: {mismatches}"
