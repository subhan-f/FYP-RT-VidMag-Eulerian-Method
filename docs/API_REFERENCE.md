# API Reference

This section lists all public functions with their parameters and return types.

## extract_frames
`extract_frames(video_path: str, fps: int = 30) -> List[np.ndarray]`
- **Parameters**: path to an MP4 and target FPS.
- **Returns**: list of RGB frames.
- **Raises**: `IOError` if the video cannot be opened.

## stabilise_frames
`stabilise_frames(frames: List[np.ndarray]) -> List[np.ndarray]`
- Feature-based global motion compensation using ORB and homography.

## crop_roi
`crop_roi(frames: List[np.ndarray], marker_hsv=(0,255,255), pad=10) -> List[np.ndarray]`
- Crops around a red circular sticker; falls back to full frame.

## build_laplacian_pyramid / reconstruct_from_laplacian
`build_laplacian_pyramid(img: np.ndarray, levels: int = 5) -> List[np.ndarray]`
`reconstruct_from_laplacian(pyr: List[np.ndarray]) -> np.ndarray`
- Standard Laplacian/Gaussian pyramid operations.

## temporal_bandpass
`temporal_bandpass(stack: np.ndarray, fs: int, f_low: float, f_high: float, order: int = 3) -> np.ndarray`
- Butterworth band-pass filtering along time.
- Transfer function:
```
H(\omega) = \frac{1}{\sqrt{1 + (\omega/\omega_c)^{2n}}}
```

## amplify
`amplify(stack: np.ndarray, alpha: float) -> np.ndarray`
- Scales the temporal stack by `alpha`.

## magnify_motion
`magnify_motion(frames: List[np.ndarray], fs: int, f_low: float, f_high: float, alpha: float, levels: int = 5) -> List[np.ndarray]`
- Wraps the Laplacian and temporal modules to produce magnified BGR frames.

## Feature Extractors
- `energy_features(frames: List[np.ndarray]) -> np.ndarray`
- `flow_hist_features(frames: List[np.ndarray], bins: int = 16) -> np.ndarray`
- `fft_sideband_features(frames: List[np.ndarray], fs: int, f1: float) -> np.ndarray`

## build_feature_vector
`build_feature_vector(frames: List[np.ndarray], fs: int, f1: float) -> np.ndarray`
- Concatenates the three feature groups (~22 dims).

## fit_pca / transform_pca
`fit_pca(X: np.ndarray, var_ratio: float = 0.95) -> PCA`
`transform_pca(pca: PCA, X: np.ndarray) -> np.ndarray`

## train_classifier
`train_classifier(X: np.ndarray, y: np.ndarray, model_path: str)`
- Grid-search over C and gamma for an SVM-RBF; saves best model with joblib.

## infer
`infer(video_path: str, model_path: str, fs: int = 30, f1: float = 25.0) -> str`
- Runs the entire pipeline and prints the predicted label and probability.
