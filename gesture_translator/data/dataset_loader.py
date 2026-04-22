from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from gesture_translator.core.feature_builder import build_sequence_tensor


def load_samples(dataset_dir: Path, allowed_labels: set[str] | None = None) -> list[dict]:
    samples = []
    for path in sorted(dataset_dir.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = str(payload.get("label") or path.parent.name).strip()
        if allowed_labels and label not in allowed_labels:
            continue
        samples.append(
            {
                "label": label,
                "path": path,
                "sequence": payload.get("sequence", []),
                "summary": payload.get("summary", {}),
                "meta": payload.get("meta", {}),
            }
        )
    return samples


def stratified_split(samples: list[dict], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for sample in samples:
        buckets[sample["label"]].append(sample)

    train_samples = []
    val_samples = []
    for _, bucket in buckets.items():
        rng.shuffle(bucket)
        val_count = max(1, int(round(len(bucket) * val_ratio))) if len(bucket) > 2 else 1
        val_samples.extend(bucket[:val_count])
        train_samples.extend(bucket[val_count:])
        if not train_samples and bucket:
            train_samples.append(bucket[-1])

    return train_samples, val_samples


class GestureSequenceDataset(Dataset):
    def __init__(self, samples: list[dict], label_to_index: dict[str, int], sequence_length: int) -> None:
        self.samples = samples
        self.label_to_index = label_to_index
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        tensor = build_sequence_tensor(sample["sequence"], sequence_length=self.sequence_length)
        label_index = self.label_to_index[sample["label"]]
        return torch.tensor(tensor, dtype=torch.float32), torch.tensor(label_index, dtype=torch.long)
