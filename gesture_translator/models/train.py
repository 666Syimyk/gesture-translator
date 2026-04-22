from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from gesture_translator.config import load_labels
from gesture_translator.data.dataset_loader import GestureSequenceDataset, load_samples, stratified_split
from gesture_translator.models.gesture_classifier import GestureClassifier
from gesture_translator.utils.metrics import build_confusion_matrix, compute_accuracy, top_confusion_pairs


def evaluate_model(model, loader, device):
    model.eval()
    losses = []
    targets = []
    predictions = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            losses.append(float(loss.item()))
            predicted = torch.argmax(logits, dim=1)
            targets.extend(batch_targets.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "accuracy": compute_accuracy(targets, predictions),
        "targets": targets,
        "predictions": predictions,
    }


def _per_class_accuracy(confusion_matrix: dict) -> list[dict]:
    labels = confusion_matrix.get("labels", [])
    matrix = confusion_matrix.get("matrix", [])
    result = []
    for row_index, label in enumerate(labels):
        row = matrix[row_index] if row_index < len(matrix) else []
        total = int(sum(row))
        correct = int(row[row_index]) if row_index < len(row) else 0
        accuracy = (correct / total) if total else 0.0
        result.append(
            {
                "label": label,
                "samples": total,
                "correct": correct,
                "accuracy": round(accuracy, 4),
            }
        )
    return result


def train_gesture_model(
    dataset_dir: Path,
    artifact_dir: Path,
    labels_path: Path,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    learning_rate: float,
) -> dict:
    all_samples = load_samples(dataset_dir)
    if not all_samples:
        raise RuntimeError(f"No dataset samples found in {dataset_dir}")

    configured_labels = load_labels(labels_path)
    available_labels = [label for label in configured_labels if any(sample["label"] == label for sample in all_samples)]
    if not available_labels:
        available_labels = sorted({sample["label"] for sample in all_samples})

    filtered_samples = [sample for sample in all_samples if sample["label"] in set(available_labels)]
    train_samples, val_samples = stratified_split(filtered_samples)
    label_to_index = {label: index for index, label in enumerate(available_labels)}

    train_dataset = GestureSequenceDataset(train_samples, label_to_index, sequence_length)
    val_dataset = GestureSequenceDataset(val_samples, label_to_index, sequence_length)
    sample_tensor, _ = train_dataset[0]
    input_size = int(sample_tensor.shape[-1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureClassifier(
        input_size=input_size,
        num_classes=len(available_labels),
        hidden_size=hidden_size,
    ).to(device)

    class_counts = Counter(sample["label"] for sample in train_samples)
    class_weights = torch.tensor(
        [max(len(train_samples), 1) / max(class_counts.get(label, 1), 1) for label in available_labels],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state = None
    best_accuracy = -1.0
    history = []
    best_val_metrics = {"targets": [], "predictions": [], "loss": 0.0, "accuracy": 0.0}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_metrics = evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_metrics["accuracy"] >= best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_val_metrics = val_metrics
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
            }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state["model"], artifact_dir / "sequence_model.pt")

    confusion = build_confusion_matrix(
        targets=best_val_metrics["targets"],
        predictions=best_val_metrics["predictions"],
        labels=available_labels,
    )
    per_class_accuracy = _per_class_accuracy(confusion)
    confusion_pairs = top_confusion_pairs(confusion, top_k=8)

    metadata = {
        "labels": available_labels,
        "sequence_length": int(sequence_length),
        "input_size": input_size,
        "hidden_size": int(hidden_size),
        "best_epoch": int(best_state["epoch"]),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "dataset_dir": str(dataset_dir),
        "class_weights": {label: round(float(weight), 4) for label, weight in zip(available_labels, class_weights.tolist())},
    }
    metrics = {
        "history": history,
        "best_accuracy": best_accuracy,
        "confusion_matrix": confusion,
        "per_class_accuracy": per_class_accuracy,
        "most_confused_pairs": confusion_pairs,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "artifact_dir": str(artifact_dir),
        "best_accuracy": round(best_accuracy, 4),
        "labels": available_labels,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "per_class_accuracy": per_class_accuracy,
        "most_confused_pairs": confusion_pairs,
    }
