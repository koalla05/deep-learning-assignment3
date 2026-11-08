"""Scores built on the competition metric, reported per language and as a competition mix."""

from eda.core.analysis import competition_f1, published_normalize

# 428 of the 477 test posts contain Ukrainian-specific letters.
COMPETITION_MIX = {"uk": 0.9, "ru": 0.1}


def f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return dict(precision=precision, recall=recall, f1=f1)


def gold_strings(row):
    return [row["text"][a:b] for a, b in row["markers"]]


def evaluate(rows, predictions):
    """F1 per language, pooled, and `mix`, where each language counts as in the test set.

    `mix` scales each language's counts per row before pooling, so a language with
    more rows does not get extra weight.
    """
    result = {}
    counts = {}
    for source in sorted({row["source"] for row in rows}):
        pairs = [(gold_strings(row), predicted) for row, predicted in zip(rows, predictions)
                 if row["source"] == source]
        score = competition_f1([p[0] for p in pairs], [p[1] for p in pairs])
        counts[source] = (score["tp"], score["fp"], score["fn"], len(pairs))
        result[source] = f1_from_counts(score["tp"], score["fp"], score["fn"])
    pooled = [sum(counts[source][i] for source in counts) for i in range(3)]
    result["pooled"] = f1_from_counts(*pooled)
    if set(counts) >= set(COMPETITION_MIX):
        mixed = [sum(share * counts[source][i] / counts[source][3]
                     for source, share in COMPETITION_MIX.items()) for i in range(3)]
        result["mix"] = f1_from_counts(*mixed)
    else:
        result["mix"] = result["pooled"]
    return result


def unseen_recall(rows, predictions, seen):
    """Recall on gold names never labeled in the rows the model trained on, per language."""
    found = {}
    for row, predicted in zip(rows, predictions):
        guesses = {published_normalize(value) for value in predicted}
        for value in {published_normalize(value) for value in gold_strings(row)}:
            if value and value not in seen:
                total, correct = found.get(row["source"], (0, 0))
                found[row["source"]] = (total + 1, correct + (value in guesses))
    return {source: correct / total for source, (total, correct) in found.items()}


def seen_names(rows):
    return {published_normalize(value) for row in rows for value in gold_strings(row)}
