import pytest


@pytest.mark.parametrize(
    "answer",
    [
        "Unused items can be returned within 30 days.",
        "You can return items after 90 days despite the 30 days rule.",
    ],
    ids=["correct-answer", "incorrect-answer"],
)
def test_keyword_check_limitation(answer):
    assert "30 days" in answer
