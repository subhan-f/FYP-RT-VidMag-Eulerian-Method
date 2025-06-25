import numpy as np
from typing import List
from sklearn.decomposition import PCA

from feat_energy import energy_features
from feat_flow import flow_hist_features
from feat_fft import fft_sideband_features


def build_feature_vector(frames: List[np.ndarray], fs: int, f1: float) -> np.ndarray:
    """Compute and concatenate energy, flow histogram and FFT sideband features."""
    energy = energy_features(frames)
    flow = flow_hist_features(frames)
    fft = fft_sideband_features(frames, fs, f1)
    return np.concatenate([energy, flow, fft]).astype(np.float32)


def fit_pca(X: np.ndarray, var_ratio: float = 0.95) -> PCA:
    """Fit PCA on data retaining ``var_ratio`` of variance."""
    pca = PCA(n_components=var_ratio)
    pca.fit(X)
    return pca


def transform_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    """Apply a trained PCA to new data."""
    return pca.transform(X)


if __name__ == "__main__":
    # Demo with random frames
    rng = np.random.default_rng(1)
    dataset = []
    for _ in range(10):
        frames = [(rng.random((64, 64, 3)) * 255).astype(np.uint8) for _ in range(5)]
        feat = build_feature_vector(frames, fs=30, f1=5.0)
        dataset.append(feat)
    X = np.vstack(dataset)
    print("Feature matrix shape:", X.shape)
    pca = fit_pca(X, var_ratio=0.9)
    X_pca = transform_pca(pca, X)
    print("PCA reduced shape:", X_pca.shape)
