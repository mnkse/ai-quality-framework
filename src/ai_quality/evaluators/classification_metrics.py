def calculate_metrics(labels, predictions):
    labels = list(labels)
    predictions = list(predictions)

    if not labels or len(labels) != len(predictions):
        raise ValueError("Provide equally sized, non-empty inputs.")

    if any(type(value) is not bool for value in labels + predictions):
        raise ValueError("Labels and predictions must be booleans.")

    tp = sum(label and prediction
             for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction
             for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction
             for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction
             for label, prediction in zip(labels, predictions))

    def divide(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": divide(tp + tn, len(labels)),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
    }
