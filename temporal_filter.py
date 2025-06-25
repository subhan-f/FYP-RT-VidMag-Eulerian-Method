import numpy as np
from scipy.signal import butter, filtfilt


def temporal_bandpass(stack: np.ndarray, fs: int, f_low: float, f_high: float, order: int = 3) -> np.ndarray:
    """Apply a Butterworth band-pass filter along the time axis."""
    nyq = 0.5 * fs
    low = f_low / nyq
    high = f_high / nyq
    b, a = butter(order, [low, high], btype="band")
    T = stack.shape[0]
    flat = stack.reshape(T, -1)
    filtered = filtfilt(b, a, flat, axis=0)
    return filtered.reshape(stack.shape)


def amplify(stack: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly scale the stack by ``alpha``."""
    return stack * alpha


if __name__ == "__main__":
    fs = 30  # Hz
    t = np.arange(0, 2, 1 / fs)
    freq = 3.0  # target frequency
    clean = np.sin(2 * np.pi * freq * t)
    noisy = clean + 0.5 * np.random.randn(len(t))
    stack = noisy[:, None, None].astype(np.float32)

    filtered = temporal_bandpass(stack, fs, 2.5, 3.5).squeeze()

    fft_freqs = np.fft.rfftfreq(len(t), 1 / fs)
    orig_fft = np.abs(np.fft.rfft(noisy))
    filt_fft = np.abs(np.fft.rfft(filtered))

    orig_peak = fft_freqs[np.argmax(orig_fft)]
    filt_peak = fft_freqs[np.argmax(filt_fft)]

    target_idx = np.argmin(np.abs(fft_freqs - freq))
    amp_before = orig_fft[target_idx]
    amp_after = filt_fft[target_idx]

    noise_idx = np.argmin(np.abs(fft_freqs - 10.0))
    noise_before = orig_fft[noise_idx]
    noise_after = filt_fft[noise_idx]

    print(f"Original peak frequency: {orig_peak:.2f} Hz")
    print(f"Filtered peak frequency: {filt_peak:.2f} Hz")
    print(f"Amplitude at {freq:.1f} Hz before: {amp_before:.3f}")
    print(f"Amplitude at {freq:.1f} Hz after : {amp_after:.3f}")
    print(f"Noise component at 10 Hz before: {noise_before:.3f}")
    print(f"Noise component at 10 Hz after : {noise_after:.3f}")

