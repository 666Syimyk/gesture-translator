import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PHRASE_DIR = Path(__file__).resolve().parent
ML_DIR = PHRASE_DIR.parent
if str(PHRASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHRASE_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config import load_config, resolve_backend_path
from augmentation import augment_samples
from data import build_class_entries, build_label_to_index, load_dataset_splits
from train_sequence_model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    SequenceDataset,
    build_sequence_tensors,
    create_torch_model,
    evaluate_torch_classifier,
    save_artifacts,
    set_seed,
)


def tensor_loader(sequences, labels, lengths, batch_size, shuffle):
    dataset = SequenceDataset(sequences, labels, lengths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model, sequences, labels, lengths, batch_size, criterion):
    if sequences.size == 0:
        return None

    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_sequences, batch_labels, batch_lengths in tensor_loader(
            sequences,
            labels,
            lengths,
            batch_size,
            shuffle=False,
        ):
            logits = model(batch_sequences, batch_lengths)
            loss = criterion(logits, batch_labels)
            total_loss += float(loss.item()) * int(batch_labels.shape[0])
            total_count += int(batch_labels.shape[0])

    return total_loss / max(total_count, 1)


def train_with_early_stopping(
    model,
    train_sequences,
    train_labels,
    train_lengths,
    val_sequences,
    val_labels,
    val_lengths,
    *,
    batch_size,
    learning_rate,
    epochs,
    patience,
):
    class_count = int(model.classifier.out_features)
    class_counts = np.bincount(train_labels, minlength=class_count).astype(np.float32)
    class_counts[class_counts == 0.0] = 1.0
    class_weights = (1.0 / class_counts) * float(np.mean(class_counts))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loader = tensor_loader(
        train_sequences,
        train_labels,
        train_lengths,
        int(batch_size),
        shuffle=True,
    )

    history = []
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_left = int(patience)
    use_validation = bool(val_sequences.size)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch_sequences, batch_labels, batch_lengths in loader:
            optimizer.zero_grad()
            logits = model(batch_sequences, batch_lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * int(batch_labels.shape[0])
            train_count += int(batch_labels.shape[0])

        train_loss = train_loss / max(train_count, 1)
        val_loss = evaluate_loss(
            model,
            val_sequences,
            val_labels,
            val_lengths,
            int(batch_size),
            criterion,
        )
        monitor_loss = val_loss if use_validation else train_loss
        improved = monitor_loss < best_loss - 1e-5

        if improved:
            best_loss = monitor_loss
            best_state = deepcopy(model.state_dict())
            patience_left = int(patience)
        else:
            patience_left -= 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_loss), 6) if val_loss is not None else None,
                "monitor_loss": round(float(monitor_loss), 6),
                "improved": improved,
            }
        )

        if use_validation and patience_left <= 0:
            break

    model.load_state_dict(best_state)
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PHRASE_DIR / "default_config.json"))
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--confidence-threshold", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]

    if args.epochs is not None:
        model_config["epochs"] = args.epochs
    if args.hidden_size is not None:
        model_config["hidden_size"] = args.hidden_size
    if args.sequence_length is not None:
        config["sequence_length"] = args.sequence_length

    dataset_dir = resolve_backend_path(args.dataset_dir or config["dataset_dir"])
    output_dir = resolve_backend_path(args.output_dir or config["artifacts_dir"])
    labels = config["phrases"]
    recognition_level = config.get("recognition_level", "phrase")
    feature_mode = config.get("feature_mode", "full")
    sequence_length = int(config.get("sequence_length", 32))
    confidence_threshold = float(
        args.confidence_threshold
        if args.confidence_threshold is not None
        else config["inference"].get(
            "confidence_threshold",
            DEFAULT_CONFIDENCE_THRESHOLD,
        )
    )

    set_seed(int(model_config.get("seed", 42)))
    splits = load_dataset_splits(dataset_dir, labels, recognition_level=recognition_level)
    augmentation_config = config.get("augmentation", {})
    augmented_train_samples = augment_samples(
        splits["train"],
        augmentation_config,
        seed=int(model_config.get("seed", 42)),
    )
    if augmented_train_samples:
        splits["train"] = [*splits["train"], *augmented_train_samples]

    class_entries = build_class_entries(labels, recognition_level=recognition_level)
    label_to_index = build_label_to_index(class_entries)

    if not splits["train"]:
        raise SystemExit(
            f"No training samples found in {dataset_dir / 'train'}. Record dataset clips first."
        )

    train_sequences, train_labels, train_lengths = build_sequence_tensors(
        splits["train"],
        label_to_index,
        sequence_length,
        feature_mode=feature_mode,
    )
    val_sequences, val_labels, val_lengths = build_sequence_tensors(
        splits["val"],
        label_to_index,
        sequence_length,
        feature_mode=feature_mode,
    )
    test_sequences, test_labels, test_lengths = build_sequence_tensors(
        splits["test"],
        label_to_index,
        sequence_length,
        feature_mode=feature_mode,
    )

    input_size = int(train_sequences.shape[2])
    model_type = str(model_config.get("type", "lstm")).lower()
    model = create_torch_model(
        model_type,
        input_size,
        int(model_config.get("hidden_size", 128)),
        len(class_entries),
    )
    model, history = train_with_early_stopping(
        model,
        train_sequences,
        train_labels,
        train_lengths,
        val_sequences,
        val_labels,
        val_lengths,
        batch_size=int(model_config.get("batch_size", 16)),
        learning_rate=float(model_config.get("learning_rate", 0.001)),
        epochs=int(model_config.get("epochs", 80)),
        patience=int(model_config.get("early_stopping_patience", 8)),
    )

    metrics = {
        "train": evaluate_torch_classifier(
            model,
            train_sequences,
            train_labels,
            train_lengths,
            class_entries,
            confidence_threshold=confidence_threshold,
        ),
        "val": evaluate_torch_classifier(
            model,
            val_sequences,
            val_labels,
            val_lengths,
            class_entries,
            confidence_threshold=confidence_threshold,
        ),
        "test": evaluate_torch_classifier(
            model,
            test_sequences,
            test_labels,
            test_lengths,
            class_entries,
            confidence_threshold=confidence_threshold,
        ),
    }

    dataset_summary = {
        "dataset_dir": str(dataset_dir),
        "train_original_count": len(splits["train"]) - len(augmented_train_samples),
        "train_count": len(splits["train"]),
        "train_augmented_count": len(augmented_train_samples),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "label_count": len(class_entries),
        "recognition_level": recognition_level,
        "feature_mode": feature_mode,
        "sequence_length": sequence_length,
        "storage": "class_folders_json_landmarks",
    }
    evaluation_summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "confidence_threshold": confidence_threshold,
        "splits": metrics,
        "labels": [entry["text"] for entry in class_entries],
        "label_keys": [entry["key"] for entry in class_entries],
        "class_entries": class_entries,
    }

    metadata = save_artifacts(
        output_dir,
        {
            "model_state_dict": model.state_dict(),
            "hidden_size": int(model_config.get("hidden_size", 128)),
            "max_sequence_length": sequence_length,
            "model_type": model_type,
            "class_entries": class_entries,
            "input_size": input_size,
            "metrics": metrics,
            "dataset_summary": dataset_summary,
            "evaluation": evaluation_summary,
            "config": {
                "model_type": model_type,
                "epochs": int(model_config.get("epochs", 80)),
                "trained_epochs": len(history),
                "batch_size": int(model_config.get("batch_size", 16)),
                "learning_rate": float(model_config.get("learning_rate", 0.001)),
                "early_stopping_patience": int(
                    model_config.get("early_stopping_patience", 8)
                ),
                "max_sequence_length": sequence_length,
                "hidden_size": int(model_config.get("hidden_size", 128)),
                "confidence_threshold": confidence_threshold,
                "seed": int(model_config.get("seed", 42)),
                "training_history": history,
                "augmentation": {
                    **augmentation_config,
                    "created_samples": len(augmented_train_samples),
                },
                "source_config": config,
            },
            "recognition_level": recognition_level,
            "feature_mode": feature_mode,
        },
    )

    print(
        json.dumps(
            {
                "ok": True,
                "outputDir": str(output_dir),
                "metadataPath": str(output_dir / "metadata.json"),
                "weightsPath": str(output_dir / metadata["artifacts"]["weights"]),
                "evaluationPath": str(output_dir / "evaluation.json"),
                "modelType": metadata["model_type"],
                "trainedEpochs": len(history),
                "metrics": metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
