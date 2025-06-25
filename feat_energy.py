import cv2
import numpy as np
from typing import List


def energy_features(frames: List[np.ndarray]) -> np.ndarray:
    """Compute mean/std/max of absolute frame differences.

    Parameters
    ----------
    frames : List[np.ndarray]
        Sequence of RGB images.

    Returns
    -------
    np.ndarray
        Vector of shape ``(3,)`` with [mean, std, max].
    """
    if len(frames) < 2:
        return np.zeros(3, dtype=np.float32)

    diffs = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        diff = np.abs(gray - prev_gray)
        diffs.append(diff)
        prev_gray = gray

    diffs = np.stack(diffs, axis=0)
    per_frame_mean = diffs.mean(axis=(1, 2))
    per_frame_std = diffs.std(axis=(1, 2))
    per_frame_max = diffs.max(axis=(1, 2))

    feat = np.array([
        per_frame_mean.mean(),
        per_frame_std.mean(),
        per_frame_max.mean(),
    ], dtype=np.float32)
    return feat


if __name__ == "__main__":
    # simple test on random frames
    rng = np.random.default_rng(0)
    frames = [(rng.random((64, 64, 3)) * 255).astype(np.uint8) for _ in range(5)]
    fvec = energy_features(frames)
    print("Energy features:", fvec)
