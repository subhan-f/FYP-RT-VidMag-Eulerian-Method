import cv2
import numpy as np
from typing import List


def flow_hist_features(frames: List[np.ndarray], bins: int = 16) -> np.ndarray:
    """Compute averaged optical-flow magnitude histogram over a sequence.

    Parameters
    ----------
    frames : List[np.ndarray]
        RGB frames.
    bins : int, optional
        Number of histogram bins, by default 16.

    Returns
    -------
    np.ndarray
        Histogram of shape ``(bins,)``.
    """
    if len(frames) < 2:
        return np.zeros(bins, dtype=np.float32)

    target_h = 160
    resized = [cv2.resize(f, (int(f.shape[1] * target_h / f.shape[0]), target_h))
               for f in frames]

    mags = []
    max_mag = 0.0
    for i in range(len(resized) - 1):
        g1 = cv2.cvtColor(resized[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(resized[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.linalg.norm(flow, axis=2)
        mags.append(mag)
        max_mag = max(max_mag, float(mag.max()))

    max_mag = max(max_mag, 1e-3)
    hist_accum = np.zeros(bins, dtype=np.float32)
    for mag in mags:
        hist, _ = np.histogram(mag, bins=bins, range=(0, max_mag))
        hist_accum += hist.astype(np.float32) / mag.size
    hist_mean = hist_accum / len(mags)
    return hist_mean


if __name__ == "__main__":
    # Simple demo with a moving square
    frames = []
    h, w = 120, 160
    for i in range(5):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (20 + i*2, 40), (40 + i*2, 60), (255, 255, 255), -1)
        frames.append(img)
    feat = flow_hist_features(frames, bins=8)
    print("Flow histogram:", feat)
