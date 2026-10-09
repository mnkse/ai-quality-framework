import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ai_quality.evaluators.llm_judge import AnswerEvaluator


pytestmark = pytest.mark.offline


def make_evaluator(output_text):
    evaluator = AnswerEvaluator.__new__(AnswerEvaluator)
    evaluator.model = "test-model"
    evaluator.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=Mock(
                return_value=SimpleNamespace(output_text=output_text)
            )
        )
    )
    return evaluator


@pytest.mark.parametrize(
    "output_text",
    [
        "not JSON",
        "",
        "[]",
        '{"passed": true}',
        '{"passed": "false", "reason": "Wrong answer."}',
        '{"passed": 1, "reason": "Correct answer."}',
        '{"passed": true, "reason": ""}',
        '{"passed": true, "reason": null}',
        '{"passed": true, "reason": "OK", "extra": 123}',
    ],
    ids=[
        "invalid-json", "empty", "wrong-container", "missing-field",
        "string-decision", "numeric-decision", "empty-reason",
        "null-reason", "extra-field",
    ],
)
def test_judge_rejects_invalid_output(output_text):
    evaluator = make_evaluator(output_text)

    with pytest.raises(ValueError):
        evaluator.evaluate("Question", "Answer", "Reference")


@pytest.mark.parametrize("passed", [True, False])
def test_judge_accepts_valid_output(passed):
    expected = {"passed": passed, "reason": "Evaluation explanation."}
    evaluator = make_evaluator(json.dumps(expected))

    result = evaluator.evaluate("Question", "Answer", "Reference")

    assert result == expected
    evaluator.client.responses.create.assert_called_once()
