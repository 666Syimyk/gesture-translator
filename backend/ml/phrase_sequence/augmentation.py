import copy

import numpy as np


LANDMARK_KEYS = (
    "left_hand",
    "right_hand",
    "face",
    "pose",
    "left_hand_world",
    "right_hand_world",
    "pose_world",
)


def jitter_point(point, rng, noise_std, scale, shift):
    augmented = dict(point)
    for axis in ("x", "y", "z"):
        if axis in augmented:
            value = float(augmented.get(axis, 0.0))
            augmented[axis] = round(
                (value * scale) + shift + float(rng.normal(0.0, noise_std)),
                6,
            )
    return augmented


def augment_frame(frame, rng, config):
    scale = 1.0 + float(rng.normal(0.0, config["scale_jitter_std"]))
    shift = float(rng.normal(0.0, config["shift_jitter_std"]))
    noise_std = float(config["coordinate_noise_std"])
    augmented = copy.deepcopy(frame)

    for key in LANDMARK_KEYS:
        points = augmented.get(key)
        if not points:
            continue
        augmented[key] = [
            jitter_point(point, rng, noise_std, scale, shift)
            for point in points
        ]

    return augmented


def augment_sequence(frames, rng, config):
    drop_probability = float(config["frame_drop_probability"])
    kept_frames = [
        augment_frame(frame, rng, config)
        for frame in frames
        if len(frames) <= 2 or rng.random() >= drop_probability
    ]

    if not kept_frames:
        kept_frames = [augment_frame(frame, rng, config) for frame in frames[:1]]

    for index, frame in enumerate(kept_frames):
        frame["frame_index"] = index

    return kept_frames


def augment_samples(samples, config, seed=42):
    if not config.get("enabled", False):
        return []

    copies_per_sample = int(config.get("copies_per_sample", 0) or 0)
    if copies_per_sample <= 0:
        return []

    normalized_config = {
        "coordinate_noise_std": float(config.get("coordinate_noise_std", 0.01)),
        "scale_jitter_std": float(config.get("scale_jitter_std", 0.03)),
        "shift_jitter_std": float(config.get("shift_jitter_std", 0.015)),
        "frame_drop_probability": float(config.get("frame_drop_probability", 0.05)),
    }
    rng = np.random.default_rng(int(seed))
    augmented_samples = []

    for sample in samples:
        for copy_index in range(copies_per_sample):
            augmented = copy.deepcopy(sample)
            augmented["sequence"] = augment_sequence(
                sample.get("sequence", []),
                rng,
                normalized_config,
            )
            augmented["meta"] = {
                **(sample.get("meta") or {}),
                "augmented": True,
                "augmentation_copy_index": copy_index,
            }
            augmented_samples.append(augmented)

    return augmented_samples

