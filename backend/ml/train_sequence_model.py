import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


HAND_POINTS = list(range(21))
POSE_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
FACE_POINTS = [1, 33, 61, 199, 263, 291]
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_MAX_SEQUENCE_LENGTH = 48
DEFAULT_EPOCHS = 18
DEFAULT_BATCH_SIZE = 16
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_MODEL_TYPE = "baseline"
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_CLASS_BALANCE = "none"
DEFAULT_CLASS_WEIGHT_POWER = 0.5
DEFAULT_FOCUS_WEIGHT_MULTIPLIER = 1.0
SUPPORTED_MODEL_TYPES = ("baseline", "gru", "lstm", "tcn")
SUPPORTED_CLASS_BALANCE = ("none", "loss", "sampler", "both")
SUPPORTED_RECOGNITION_LEVELS = ("alphabet", "sign", "phrase")
SEED = 42

ALPHABET_UNIT_TEXT = {
    "NUMBER_0": "0",
    "NUMBER_1": "1",
    "NUMBER_2": "2",
    "NUMBER_3": "3",
    "NUMBER_4": "4",
    "NUMBER_5": "5",
    "NUMBER_6": "6",
    "NUMBER_7": "7",
    "NUMBER_8": "8",
    "NUMBER_9": "9",
    "LETTER_A": "А",
    "LETTER_B": "Б",
    "LETTER_V": "В",
    "LETTER_G": "Г",
    "LETTER_D": "Д",
    "LETTER_E": "Е",
    "LETTER_YO": "Ё",
    "LETTER_ZH": "Ж",
    "LETTER_Z": "З",
    "LETTER_I": "И",
    "LETTER_I_SHORT": "Й",
    "LETTER_K": "К",
    "LETTER_L": "Л",
    "LETTER_M": "М",
    "LETTER_N": "Н",
    "LETTER_O": "О",
    "LETTER_P": "П",
    "LETTER_R": "Р",
    "LETTER_S": "С",
    "LETTER_T": "Т",
    "LETTER_U": "У",
    "LETTER_F": "Ф",
    "LETTER_KH": "Х",
    "LETTER_TS": "Ц",
    "LETTER_CH": "Ч",
    "LETTER_SH": "Ш",
    "LETTER_SHCH": "Щ",
    "LETTER_HARD_SIGN": "Ъ",
    "LETTER_YERU": "Ы",
    "LETTER_SOFT_SIGN": "Ь",
    "LETTER_EH": "Э",
    "LETTER_YU": "Ю",
    "LETTER_YA": "Я",
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def parse_csv_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split(",")

    return [str(item).strip() for item in items if str(item).strip()]


def normalize_recognition_level(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SUPPORTED_RECOGNITION_LEVELS:
            return normalized

    return "phrase"


def normalize_display_text(text, recognition_level, unit_code=None):
    normalized_text = str(text or "").strip()
    normalized_unit_code = str(unit_code or "").strip()

    if recognition_level == "alphabet":
        resolved = ALPHABET_UNIT_TEXT.get(normalized_unit_code)
        if resolved:
            return resolved

    return normalized_text or normalized_unit_code


def normalize_class_entry(entry):
    recognition_level = normalize_recognition_level(entry.get("recognition_level"))
    unit_code = str(entry.get("unit_code") or "").strip()
    text = normalize_display_text(entry.get("text"), recognition_level, unit_code)
    identifier = unit_code or text or f"class_{recognition_level}"

    return {
        "key": str(entry.get("key") or f"{recognition_level}::{identifier}"),
        "text": text or identifier,
        "recognition_level": recognition_level,
        "unit_code": unit_code or None,
    }


def infer_dataset_recognition_level(*sample_groups):
    levels = {
        normalize_recognition_level(sample.get("recognition_level"))
        for samples in sample_groups
        for sample in samples
    }

    if len(levels) == 1:
        return levels.pop()

    return "unified"


def resolve_feature_mode(recognition_level):
    return "hands_only" if recognition_level == "alphabet" else "full"


def infer_feature_mode(*sample_groups):
    has_face_or_pose = False
    has_hand = False

    for samples in sample_groups:
        for sample in samples:
            for frame in sample.get("sequence", []):
                if frame.get("left_hand") or frame.get("right_hand"):
                    has_hand = True
                if frame.get("face") or frame.get("pose"):
                    has_face_or_pose = True
                    break
            if has_face_or_pose:
                break
        if has_face_or_pose:
            break

    if has_face_or_pose:
        return "full"

    if has_hand:
        return "hands_only"

    return None


def build_class_entry(sample):
    recognition_level = normalize_recognition_level(sample.get("recognition_level"))
    phrase_text = str(sample.get("phrase_text") or "").strip()
    unit_code = str(sample.get("unit_code") or "").strip()
    text = normalize_display_text(phrase_text, recognition_level, unit_code)
    identifier = unit_code or text or f"class_{recognition_level}"

    return normalize_class_entry(
        {
            "key": f"{recognition_level}::{identifier}",
            "text": text or identifier,
            "recognition_level": recognition_level,
            "unit_code": unit_code or None,
        }
    )


def resolve_focus_label_keys(class_entries, focus_labels):
    requested = {str(label).strip() for label in focus_labels if str(label).strip()}
    if not requested:
        return []

    resolved = []
    for entry in class_entries:
        candidates = {
            entry["key"],
            str(entry.get("unit_code") or "").strip(),
            str(entry.get("text") or "").strip(),
        }
        if requested.intersection({value for value in candidates if value}):
            resolved.append(entry["key"])

    return resolved


def safe_point(points, index, include_visibility=False):
    if index >= len(points):
        return [0.0, 0.0, 0.0, 0.0] if include_visibility else [0.0, 0.0, 0.0]

    point = points[index]
    values = [
        float(point.get("x", 0.0)),
        float(point.get("y", 0.0)),
        float(point.get("z", 0.0)),
    ]

    if include_visibility:
        values.append(float(point.get("visibility", 0.0)))

    return values


def get_primary_hand_label(frame):
    handedness = frame.get("handedness", []) or []

    if handedness:
        best = max(
            handedness,
            key=lambda item: float(item.get("score", 0.0) or 0.0),
        )
        label = str(best.get("label", "")).strip().lower()
        if label in {"left", "right"}:
            return label

    left_points = frame.get("left_hand", []) or []
    right_points = frame.get("right_hand", []) or []

    if len(right_points) > len(left_points):
        return "right"
    if len(left_points) > len(right_points):
        return "left"

    return None


def resolve_sequence_primary_hand(frames):
    left_votes = 0
    right_votes = 0

    for frame in frames:
        label = get_primary_hand_label(frame)
        if label == "left":
            left_votes += 1
        elif label == "right":
            right_votes += 1

    if right_votes > left_votes:
        return "right"
    if left_votes > right_votes:
        return "left"

    return "right"


def get_hand_points_for_label(frame, preferred_label):
    preferred = frame.get(f"{preferred_label}_hand", []) or []
    alternate_label = "left" if preferred_label == "right" else "right"
    alternate = frame.get(f"{alternate_label}_hand", []) or []

    if preferred:
        return preferred, preferred_label
    if alternate:
        return alternate, alternate_label

    return [], preferred_label


def normalize_hand_points(points, actual_label):
    if not points:
        return [[0.0, 0.0, 0.0] for _ in HAND_POINTS]

    wrist = safe_point(points, 0)
    middle_mcp = safe_point(points, 9)
    index_mcp = safe_point(points, 5)
    pinky_mcp = safe_point(points, 17)

    translated_points = []
    for index in HAND_POINTS:
        x, y, z = safe_point(points, index)
        x -= wrist[0]
        y -= wrist[1]
        z -= wrist[2]
        if actual_label == "left":
            x = -x
        translated_points.append([x, y, z])

    mirrored_middle = translated_points[9]
    mirrored_index = translated_points[5]
    mirrored_pinky = translated_points[17]
    palm_axis_scale = np.linalg.norm(np.asarray(mirrored_middle[:2]))
    palm_width_scale = np.linalg.norm(
        np.asarray(mirrored_index[:2]) - np.asarray(mirrored_pinky[:2])
    )
    scale = max(palm_axis_scale, palm_width_scale, 1e-4)

    angle = np.arctan2(mirrored_middle[1], mirrored_middle[0])
    target_angle = -np.pi / 2
    rotation = target_angle - angle
    cos_theta = np.cos(rotation)
    sin_theta = np.sin(rotation)

    def rotate_xy(x, y):
        return (
            x * cos_theta - y * sin_theta,
            x * sin_theta + y * cos_theta,
        )

    if scale < 1e-4:
        scale = 1.0

    normalized = []
    for x, y, z in translated_points:
        rotated_x, rotated_y = rotate_xy(x, y)
        normalized.append([rotated_x / scale, rotated_y / scale, z / scale])

    return normalized


def normalize_frame(frame_vector):
    array = np.asarray(frame_vector, dtype=np.float32)

    if not array.size:
        return array

    non_zero = array[np.nonzero(array)]
    scale = np.max(np.abs(non_zero)) if non_zero.size else 1.0
    if scale <= 0:
        scale = 1.0

    return array / scale


def extract_frame_vector(frame, feature_mode="full", primary_hand_label=None):
    vector = []

    if feature_mode == "hands_only":
        hand_points, actual_label = get_hand_points_for_label(
            frame,
            primary_hand_label or "right",
        )
        normalized_points = normalize_hand_points(hand_points, actual_label)
        for point in normalized_points:
            vector.extend(point)
        return np.asarray(vector, dtype=np.float32)

    for index in HAND_POINTS:
        vector.extend(safe_point(frame.get("left_hand", []), index))

    for index in HAND_POINTS:
        vector.extend(safe_point(frame.get("right_hand", []), index))

    for index in POSE_POINTS:
        vector.extend(safe_point(frame.get("pose", []), index, include_visibility=True))

    for index in FACE_POINTS:
        vector.extend(safe_point(frame.get("face", []), index))

    return normalize_frame(vector)


def build_frame_matrix(sample, feature_mode="full"):
    frames = sample.get("sequence", [])

    if not frames:
        return np.zeros(
            (1, extract_frame_vector({}, feature_mode=feature_mode).shape[0]),
            dtype=np.float32,
        )

    primary_hand_label = (
        resolve_sequence_primary_hand(frames) if feature_mode == "hands_only" else None
    )

    return np.stack(
        [
            extract_frame_vector(
                frame,
                feature_mode=feature_mode,
                primary_hand_label=primary_hand_label,
            )
            for frame in frames
        ]
    ).astype(np.float32)


def build_sequence_feature(sample, feature_mode="full"):
    frame_matrix = build_frame_matrix(sample, feature_mode=feature_mode)

    mean_vector = frame_matrix.mean(axis=0)
    std_vector = frame_matrix.std(axis=0)
    first_vector = frame_matrix[0]
    last_vector = frame_matrix[-1]
    delta_vector = last_vector - first_vector
    motion_vector = (
        np.mean(np.abs(np.diff(frame_matrix, axis=0)), axis=0)
        if len(frame_matrix) > 1
        else np.zeros_like(mean_vector)
    )

    return np.concatenate(
        [
            mean_vector,
            std_vector,
            first_vector,
            last_vector,
            delta_vector,
            motion_vector,
        ]
    ).astype(np.float32)


def build_feature_matrix(samples, feature_mode="full"):
    if not samples:
        return np.zeros((0, 0), dtype=np.float32)

    return np.stack(
        [build_sequence_feature(sample, feature_mode=feature_mode) for sample in samples]
    ).astype(np.float32)


def build_dataset(samples, feature_mode="full"):
    if not samples:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
            {},
        )

    class_entries = sorted(
        {build_class_entry(sample)["key"]: build_class_entry(sample) for sample in samples}.values(),
        key=lambda item: (item["recognition_level"], item["text"], item["key"]),
    )
    label_to_index = {
        entry["key"]: index for index, entry in enumerate(class_entries)
    }
    features = build_feature_matrix(samples, feature_mode=feature_mode)
    y = np.asarray(
        [label_to_index[build_class_entry(sample)["key"]] for sample in samples],
        dtype=np.int64,
    )

    return features, y, class_entries, label_to_index


def standardize(train_x, *other_arrays):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0

    normalized = [(train_x - mean) / std]
    for array in other_arrays:
        normalized.append((array - mean) / std if array.size else array)

    return mean, std, normalized


def resample_or_pad_frame_matrix(frame_matrix, max_length):
    if len(frame_matrix) >= max_length:
        indices = np.linspace(0, len(frame_matrix) - 1, max_length).astype(int)
        return frame_matrix[indices], max_length

    padded = np.zeros((max_length, frame_matrix.shape[1]), dtype=np.float32)
    padded[: len(frame_matrix)] = frame_matrix
    return padded, len(frame_matrix)


def build_sequence_tensors(samples, label_to_index, max_length, feature_mode="full"):
    if not samples:
        return (
            np.zeros((0, max_length, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    matrices = []
    labels = []
    lengths = []

    for sample in samples:
        frame_matrix = build_frame_matrix(sample, feature_mode=feature_mode)
        padded_matrix, original_length = resample_or_pad_frame_matrix(
            frame_matrix, max_length
        )
        matrices.append(padded_matrix)
        labels.append(label_to_index[build_class_entry(sample)["key"]])
        lengths.append(original_length)

    return (
        np.stack(matrices).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(lengths, dtype=np.int64),
    )


def softmax(scores):
    if not scores.size:
        return scores

    shifted = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def build_confusion_matrix(labels, predictions, class_count):
    matrix = np.zeros((class_count, class_count), dtype=np.int64)

    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1

    return matrix


def build_per_class_accuracy(labels, predictions, class_entries):
    per_class = {}

    for class_index, entry in enumerate(class_entries):
        class_mask = labels == class_index
        class_count = int(np.sum(class_mask))
        class_accuracy = (
            float(np.mean(predictions[class_mask] == labels[class_mask]))
            if class_count
            else 0.0
        )

        per_class[entry["key"]] = {
            "label": entry["text"],
            "recognition_level": entry["recognition_level"],
            "unit_code": entry["unit_code"],
            "sample_count": class_count,
            "accuracy": round(class_accuracy, 4),
        }

    return per_class


def build_metrics_from_probabilities(
    probabilities,
    labels,
    class_entries,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    latency_ms_avg=0.0,
):
    if probabilities.size == 0 or labels.size == 0:
        return {
            "sample_count": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "average_confidence": 0.0,
            "low_confidence_rate": 0.0,
            "latency_ms_avg": 0.0,
            "per_class_accuracy": {},
            "confusion_matrix": [],
            "labels": [entry["text"] for entry in class_entries],
            "label_keys": [entry["key"] for entry in class_entries],
            "class_entries": class_entries,
        }

    predictions = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(predictions)), predictions]
    top_k = min(3, probabilities.shape[1])
    topk_predictions = np.argsort(probabilities, axis=1)[:, -top_k:]
    top1_accuracy = float(np.mean(predictions == labels))
    top3_accuracy = float(
        np.mean([label in topk for label, topk in zip(labels, topk_predictions)])
    )
    low_confidence_rate = float(np.mean(confidence < confidence_threshold))
    confusion_matrix = build_confusion_matrix(labels, predictions, len(class_entries))
    per_class_accuracy = build_per_class_accuracy(labels, predictions, class_entries)

    return {
        "sample_count": int(len(labels)),
        "top1_accuracy": round(top1_accuracy, 4),
        "top3_accuracy": round(top3_accuracy, 4),
        "accuracy": round(top1_accuracy, 4),
        "average_confidence": round(float(confidence.mean()), 4),
        "low_confidence_rate": round(low_confidence_rate, 4),
        "latency_ms_avg": round(float(latency_ms_avg), 4),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "labels": [entry["text"] for entry in class_entries],
        "label_keys": [entry["key"] for entry in class_entries],
        "class_entries": class_entries,
    }


def train_centroid_classifier(train_x, train_y, class_count):
    centroids = []

    for class_index in range(class_count):
        class_features = train_x[train_y == class_index]

        if class_features.size == 0:
            centroids.append(np.zeros(train_x.shape[1], dtype=np.float32))
            continue

        centroids.append(class_features.mean(axis=0))

    return np.stack(centroids).astype(np.float32)


def predict_probabilities_centroid(features, centroids):
    if not features.size:
        return np.zeros((0, centroids.shape[0]), dtype=np.float32)

    distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    return softmax((-distances).astype(np.float32))


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index], self.lengths[index]


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, class_count):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.classifier(hidden[-1])


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, class_count):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(hidden[-1])


class TemporalCNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, class_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, x, lengths):
        del lengths
        features = self.network(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(features)


def create_torch_model(model_type, input_size, hidden_size, class_count):
    if model_type == "gru":
        return GRUClassifier(input_size, hidden_size, class_count)

    if model_type == "lstm":
        return LSTMClassifier(input_size, hidden_size, class_count)

    if model_type == "tcn":
        return TemporalCNNClassifier(input_size, hidden_size, class_count)

    raise ValueError(f"Unsupported torch model type: {model_type}")


def train_torch_model(
    model_type,
    train_sequences,
    train_labels,
    train_lengths,
    input_size,
    class_count,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    class_balance=DEFAULT_CLASS_BALANCE,
    class_weight_power=DEFAULT_CLASS_WEIGHT_POWER,
    class_entries=None,
    focus_label_keys=None,
    focus_weight_multiplier=DEFAULT_FOCUS_WEIGHT_MULTIPLIER,
):
    model = create_torch_model(model_type, input_size, hidden_size, class_count)
    dataset = SequenceDataset(train_sequences, train_labels, train_lengths)
    class_counts = np.bincount(train_labels, minlength=class_count).astype(np.float32)
    class_counts[class_counts == 0.0] = 1.0
    class_weights = np.power(class_counts, -float(class_weight_power))
    focus_key_set = {
        str(key).strip() for key in (focus_label_keys or []) if str(key).strip()
    }
    focus_multiplier = float(focus_weight_multiplier or 1.0)
    if focus_key_set and focus_multiplier > 1.0 and class_entries:
        for index, entry in enumerate(class_entries):
            if entry["key"] in focus_key_set:
                class_weights[index] *= focus_multiplier

    class_weights = class_weights / float(np.mean(class_weights))
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    sampler = None
    if class_balance in {"sampler", "both"}:
        sample_weights = class_weights[np.asarray(train_labels, dtype=np.int64)]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )
    criterion = (
        nn.CrossEntropyLoss(weight=class_weight_tensor)
        if class_balance in {"loss", "both"}
        else nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for _ in range(epochs):
        for batch_sequences, batch_labels, batch_lengths in loader:
            optimizer.zero_grad()
            logits = model(batch_sequences, batch_lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    return model


def predict_probabilities_torch(model, sequences, lengths):
    if sequences.size == 0:
        class_count = model.classifier.out_features
        return np.zeros((0, class_count), dtype=np.float32)

    model.eval()

    with torch.no_grad():
        sequence_tensor = torch.tensor(sequences, dtype=torch.float32)
        length_tensor = torch.tensor(lengths, dtype=torch.long)
        logits = model(sequence_tensor, length_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    return probabilities.astype(np.float32)


def evaluate_centroid_classifier(
    features,
    labels,
    centroids,
    class_entries,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
):
    if not features.size:
        return build_metrics_from_probabilities(
            np.zeros((0, len(class_entries)), dtype=np.float32),
            labels,
            class_entries,
            confidence_threshold=confidence_threshold,
            latency_ms_avg=0.0,
        )

    start_time = perf_counter()
    probabilities = predict_probabilities_centroid(features, centroids)
    elapsed_ms = (perf_counter() - start_time) * 1000.0

    return build_metrics_from_probabilities(
        probabilities,
        labels,
        class_entries,
        confidence_threshold=confidence_threshold,
        latency_ms_avg=elapsed_ms / max(len(labels), 1),
    )


def evaluate_torch_classifier(
    model,
    sequences,
    labels,
    lengths,
    class_entries,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
):
    if not sequences.size:
        return build_metrics_from_probabilities(
            np.zeros((0, len(class_entries)), dtype=np.float32),
            labels,
            class_entries,
            confidence_threshold=confidence_threshold,
            latency_ms_avg=0.0,
        )

    start_time = perf_counter()
    probabilities = predict_probabilities_torch(model, sequences, lengths)
    elapsed_ms = (perf_counter() - start_time) * 1000.0

    return build_metrics_from_probabilities(
        probabilities,
        labels,
        class_entries,
        confidence_threshold=confidence_threshold,
        latency_ms_avg=elapsed_ms / max(len(labels), 1),
    )


def save_artifacts(output_dir, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = payload["model_type"]

    artifacts = {
        "evaluation": "evaluation.json",
    }

    if model_type == "baseline":
        np.savez_compressed(
            output_dir / "sequence_baseline_model.npz",
            centroids=payload["centroids"],
            mean=payload["mean"],
            std=payload["std"],
        )
        artifacts["weights"] = "sequence_baseline_model.npz"
    else:
        torch.save(
            {
                "state_dict": payload["model_state_dict"],
                "input_size": payload["input_size"],
                "hidden_size": payload["hidden_size"],
                "class_count": len(payload["class_entries"]),
                "max_sequence_length": payload["max_sequence_length"],
            },
            output_dir / "sequence_model.pt",
        )
        artifacts["weights"] = "sequence_model.pt"

    metadata = {
        "model_type": model_type,
        "feature_version": 3,
        "labels": [entry["text"] for entry in payload["class_entries"]],
        "label_keys": [entry["key"] for entry in payload["class_entries"]],
        "class_entries": payload["class_entries"],
        "input_size": int(payload["input_size"]),
        "recognition_level": payload["recognition_level"],
        "feature_mode": payload["feature_mode"],
        "metrics": payload["metrics"],
        "dataset": payload["dataset_summary"],
        "evaluation": payload["evaluation"],
        "config": payload["config"],
        "artifacts": artifacts,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, ensure_ascii=False, indent=2)

    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as output_file:
        json.dump(payload["evaluation"], output_file, ensure_ascii=False, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--model-type",
        choices=SUPPORTED_MODEL_TYPES,
        default=DEFAULT_MODEL_TYPE,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument(
        "--class-balance",
        choices=SUPPORTED_CLASS_BALANCE,
        default=DEFAULT_CLASS_BALANCE,
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=DEFAULT_CLASS_WEIGHT_POWER,
    )
    parser.add_argument("--focus-labels")
    parser.add_argument(
        "--focus-weight-multiplier",
        type=float,
        default=DEFAULT_FOCUS_WEIGHT_MULTIPLIER,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    train_samples = read_json(dataset_dir / "train.json")
    val_samples = read_json(dataset_dir / "val.json")
    test_samples = read_json(dataset_dir / "test.json")
    dataset_recognition_level = infer_dataset_recognition_level(
        train_samples, val_samples, test_samples
    )
    feature_mode = infer_feature_mode(train_samples, val_samples, test_samples)
    if feature_mode is None:
        feature_mode = resolve_feature_mode(dataset_recognition_level)

    train_x, train_y, class_entries, label_to_index = build_dataset(
        train_samples, feature_mode=feature_mode
    )
    requested_focus_labels = parse_csv_list(args.focus_labels)
    resolved_focus_label_keys = resolve_focus_label_keys(
        class_entries,
        requested_focus_labels,
    )

    if train_x.size == 0:
        raise SystemExit("Dataset export is empty. Prepare train/val/test first.")

    feature_size = int(train_x.shape[1])
    metrics = {}
    model_payload = {}

    if args.model_type == "baseline":
        def map_labels(samples):
            return np.asarray(
                [label_to_index[build_class_entry(sample)["key"]] for sample in samples],
                dtype=np.int64,
            )

        val_x = (
            build_feature_matrix(val_samples, feature_mode=feature_mode)
            if val_samples
            else np.zeros((0, feature_size), dtype=np.float32)
        )
        test_x = (
            build_feature_matrix(test_samples, feature_mode=feature_mode)
            if test_samples
            else np.zeros((0, feature_size), dtype=np.float32)
        )
        val_y = map_labels(val_samples) if val_samples else np.zeros((0,), dtype=np.int64)
        test_y = map_labels(test_samples) if test_samples else np.zeros((0,), dtype=np.int64)

        mean, std, normalized = standardize(train_x, val_x, test_x)
        train_x_norm, val_x_norm, test_x_norm = normalized
        centroids = train_centroid_classifier(train_x_norm, train_y, len(class_entries))

        metrics = {
            "train": evaluate_centroid_classifier(
                train_x_norm,
                train_y,
                centroids,
                class_entries,
                confidence_threshold=args.confidence_threshold,
            ),
            "val": evaluate_centroid_classifier(
                val_x_norm,
                val_y,
                centroids,
                class_entries,
                confidence_threshold=args.confidence_threshold,
            ),
            "test": evaluate_centroid_classifier(
                test_x_norm,
                test_y,
                centroids,
                class_entries,
                confidence_threshold=args.confidence_threshold,
            ),
        }

        model_payload = {
            "centroids": centroids,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }
    else:
        train_sequences, train_labels, train_lengths = build_sequence_tensors(
            train_samples,
            label_to_index,
            args.max_sequence_length,
            feature_mode=feature_mode,
        )
        val_sequences, val_labels, val_lengths = build_sequence_tensors(
            val_samples,
            label_to_index,
            args.max_sequence_length,
            feature_mode=feature_mode,
        )
        test_sequences, test_labels, test_lengths = build_sequence_tensors(
            test_samples,
            label_to_index,
            args.max_sequence_length,
            feature_mode=feature_mode,
        )

        input_size = int(train_sequences.shape[2])
        model = train_torch_model(
            args.model_type,
            train_sequences,
            train_labels,
            train_lengths,
            input_size,
            len(class_entries),
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            class_balance=args.class_balance,
            class_weight_power=args.class_weight_power,
            class_entries=class_entries,
            focus_label_keys=resolved_focus_label_keys,
            focus_weight_multiplier=args.focus_weight_multiplier,
        )

        metrics = {
            "train": evaluate_torch_classifier(
                model,
                train_sequences,
                train_labels,
                train_lengths,
                class_entries,
                confidence_threshold=args.confidence_threshold,
            ),
            "val": evaluate_torch_classifier(
                model,
                val_sequences,
                val_labels,
                val_lengths,
                class_entries,
                confidence_threshold=args.confidence_threshold,
            ),
            "test": evaluate_torch_classifier(
                model,
                test_sequences,
                test_labels,
                test_lengths,
                class_entries,
                confidence_threshold=args.confidence_threshold,
            ),
        }

        feature_size = input_size
        model_payload = {
            "model_state_dict": model.state_dict(),
            "hidden_size": args.hidden_size,
            "max_sequence_length": args.max_sequence_length,
        }

    dataset_summary = {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "label_count": len(class_entries),
        "recognition_level": dataset_recognition_level,
        "feature_mode": feature_mode,
    }

    evaluation_summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "confidence_threshold": args.confidence_threshold,
        "splits": metrics,
        "labels": [entry["text"] for entry in class_entries],
        "label_keys": [entry["key"] for entry in class_entries],
        "class_entries": class_entries,
    }

    metadata = save_artifacts(
        output_dir,
        {
            **model_payload,
            "model_type": args.model_type,
            "class_entries": class_entries,
            "input_size": feature_size,
            "metrics": metrics,
            "dataset_summary": dataset_summary,
            "evaluation": evaluation_summary,
            "config": {
                "model_type": args.model_type,
                "epochs": args.epochs,
                "max_sequence_length": args.max_sequence_length,
                "hidden_size": args.hidden_size,
                "confidence_threshold": args.confidence_threshold,
                "class_balance": args.class_balance,
                "class_weight_power": args.class_weight_power,
                "focus_labels": requested_focus_labels,
                "focus_label_keys": resolved_focus_label_keys,
                "focus_weight_multiplier": args.focus_weight_multiplier,
                "seed": args.seed,
            },
            "recognition_level": dataset_recognition_level,
            "feature_mode": feature_mode,
        },
    )

    print(
        json.dumps(
            {
                "ok": True,
                "outputDir": str(output_dir),
                "metrics": metrics,
                "evaluation": evaluation_summary,
                "labelCount": len(class_entries),
                "inputSize": feature_size,
                "metadataPath": str(output_dir / "metadata.json"),
                "weightsPath": str(output_dir / metadata["artifacts"]["weights"]),
                "evaluationPath": str(output_dir / "evaluation.json"),
                "modelType": metadata["model_type"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
