import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "test_data" / "prompts.json"

with DATA_FILE.open(encoding="utf-8-sig") as file:
    test_cases = json.load(file)


@pytest.mark.parametrize(
    "case",
    test_cases,
    ids=[case["id"] for case in test_cases],
)
def test_answer(case, ai_client):
    actual = ai_client.get_answer(case["question"])

    print(f"\nScenario: {case['id']}")
    print(f"Mode: {ai_client.mode}")
    print(f"Model: {getattr(ai_client, 'model', 'demo')}")
    print(f"Question: {case['question']}")
    print(f"Answer: {actual}")

    assert case["expected"] in actual, (
        f"Scenario: {case['id']}\n"
        f"Expected: {case['expected']}\n"
        f"Actual: {actual}"
    )
