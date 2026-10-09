import json
from pathlib import Path

import pytest

from evaluator import AnswerEvaluator
from rag_client import RAGClient


DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "test_data"
    / "rag_boundary_cases.json"
)
cases = json.loads(DATA_FILE.read_text(encoding="utf-8-sig"))


@pytest.fixture(scope="module")
def rag_client():
    return RAGClient()


@pytest.fixture(scope="module")
def evaluator():
    return AnswerEvaluator()


@pytest.mark.parametrize("case", cases, ids=[case["id"] for case in cases])
def test_rag_return_boundary(case, rag_client, evaluator):
    result = rag_client.get_answer(case["question"])

    actual_ids = {doc["id"] for doc in result["documents"]}
    assert actual_ids == {"return-policy"}, (
        f"Wrong documents retrieved: {actual_ids}"
    )

    evaluation = evaluator.evaluate(
        question=case["question"],
        answer=result["answer"],
        reference=case["reference"],
    )

    print(f"\nScenario: {case['id']}")
    print(f"Answer: {result['answer']}")
    print(f"Judge decision: {evaluation['passed']}")
    print(f"Reason: {evaluation['reason']}")

    assert evaluation["passed"], evaluation["reason"]


import pytest

pytestmark = pytest.mark.live
