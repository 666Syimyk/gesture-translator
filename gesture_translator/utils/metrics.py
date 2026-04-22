from __future__ import annotations


def compute_accuracy(targets: list[int], predictions: list[int]) -> float:
    if not targets:
        return 0.0
    correct = sum(int(target == prediction) for target, prediction in zip(targets, predictions))
    return correct / len(targets)


def build_confusion_matrix(targets: list[int], predictions: list[int], labels: list[str]) -> dict:
    matrix = [[0 for _ in labels] for _ in labels]
    for target, prediction in zip(targets, predictions):
        if 0 <= target < len(labels) and 0 <= prediction < len(labels):
            matrix[target][prediction] += 1
    return {"labels": labels, "matrix": matrix}


def top_confusion_pairs(confusion_matrix: dict, top_k: int = 8) -> list[dict]:
    labels = confusion_matrix.get("labels", [])
    matrix = confusion_matrix.get("matrix", [])
    pairs = []
    for i, source_label in enumerate(labels):
        if i >= len(matrix):
            continue
        row = matrix[i]
        for j, target_label in enumerate(labels):
            if i == j or j >= len(row):
                continue
            count = int(row[j])
            if count <= 0:
                continue
            pairs.append(
                {
                    "from": source_label,
                    "to": target_label,
                    "count": count,
                }
            )

    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[: max(1, int(top_k))]
