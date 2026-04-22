import sys
from pathlib import Path

import numpy as np


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from train_sequence_model import build_frame_matrix, resample_or_pad_frame_matrix


def frames_to_tensor(frames, sequence_length=32, feature_mode="full"):
    sample = {"sequence": frames or []}
    frame_matrix = build_frame_matrix(sample, feature_mode=feature_mode)
    padded_matrix, original_length = resample_or_pad_frame_matrix(
        frame_matrix,
        int(sequence_length),
    )
    return padded_matrix.astype(np.float32), int(original_length)


def sample_to_tensor(sample, sequence_length=32, feature_mode="full"):
    return frames_to_tensor(
        sample.get("sequence", []),
        sequence_length=sequence_length,
        feature_mode=feature_mode,
    )

