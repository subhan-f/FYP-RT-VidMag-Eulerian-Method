import cv2
import numpy as np
from typing import List


def fft_sideband_features(frames: List[np.ndarray], fs: int, f1: float) -> np.ndarray:
    """Compute FFT sideband amplitudes from mean-intensity signal.

    Parameters
    ----------
    frames : List[np.ndarray]
        RGB frames of the ROI.
    fs : int
        Frame rate of the sequence.
    f1 : float
        Fundamental frequency to probe.

    Returns
    -------
    np.ndarray
        Array ``(3,)`` with [A1, A2, A2/A1].
    """
    if not frames:
        return np.zeros(3, dtype=np.float32)

    signal = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32).mean() for f in frames]
    signal = np.asarray(signal, dtype=np.float32)

    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal))

    def amp_at(freq: float) -> float:
        idx = np.argmin(np.abs(freqs - freq))
        return float(spectrum[idx])

    A1 = amp_at(f1)
    A2 = amp_at(2 * f1)
    ratio = A2 / A1 if A1 > 1e-8 else 0.0

    return np.array([A1, A2, ratio], dtype=np.float32)


if __name__ == "__main__":
    fs = 30
    f1 = 4.0
    t = np.arange(0, 1, 1 / fs)
    # synthetic intensity signal with fundamental and second harmonic
    sig = 0.8 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * 2 * f1 * t)
    frames = [np.full((64, 64, 3), 128 + int(50 * s), dtype=np.uint8) for s in sig]
    feats = fft_sideband_features(frames, fs, f1)
    print("Sideband features:", feats)
