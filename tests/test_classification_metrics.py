import pytest

from ai_quality.evaluators.classification_metrics import calculate_metrics


pytestmark = pytest.mark.offline


def test_metrics_for_imbalanced_results():
    # TP=2, FP=1, TN=3, FN=1
    result = calculate_metrics(
        [True, True, True, False, False, False, False],
        [True, True, False, True, False, False, False],
    )

    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (
        2, 1, 3, 1
    )
    assert result["accuracy"] == pytest.approx(5 / 7)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)


def test_no_positive_predictions():
    result = calculate_metrics(
        [True, False],
        [False, False],
    )

    assert result["precision"] is None
    assert result["recall"] == 0
    assert result["f1"] == 0


def test_no_actual_positives():
    result = calculate_metrics(
        [False, False],
        [True, False],
    )

    assert result["precision"] == 0
    assert result["recall"] is None
    assert result["f1"] == 0


def test_only_true_negatives():
    result = calculate_metrics(
        [False, False],
        [False, False],
    )

    assert result["accuracy"] == 1
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None


@pytest.mark.parametrize(
    "labels, predictions",
    [
        ([], []),
        ([True], [True, False]),
        ([1], [True]),
        ([True], ["True"]),
    ],
    ids=["empty", "different-lengths", "invalid-label", "invalid-prediction"],
)
def test_invalid_inputs(labels, predictions):
    with pytest.raises(ValueError):
        calculate_metrics(labels, predictions)
