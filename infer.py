import argparse
import cv2
import numpy as np
from joblib import load

from extract_frames import extract_frames
from stabilise_frames import stabilise_frames
from crop_roi import crop_roi
from magnify_motion import magnify_motion
from build_features import build_feature_vector


def infer(video_path: str, model_path: str, fs: int = 30, f1: float = 25.0) -> str:
    """Run the full pipeline on ``video_path`` and return class label."""
    frames = extract_frames(video_path, fs)
    if not frames:
        raise ValueError("No frames extracted")

    frames = stabilise_frames(frames)
    frames = crop_roi(frames)

    magnified_bgr = magnify_motion(frames, fs, f1 - 1.0, f1 + 1.0, alpha=10.0, levels=5)
    magnified_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in magnified_bgr]

    feat = build_feature_vector(magnified_rgb, fs, f1).reshape(1, -1)

    model = load(model_path)

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(feat)[0, 1])
    else:
        dec = float(model.decision_function(feat)[0])
        proba = 1.0 / (1.0 + np.exp(-dec))

    pred = int(model.predict(feat)[0])
    label = "Normal" if pred == 0 else "Abnormal"
    print(f"Prediction: {label} (prob={proba:.3f})")
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a video")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("model_path", help="Path to trained model (.joblib)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--f1", type=float, default=25.0, help="Fundamental frequency")
    args = parser.parse_args()

    infer(args.video_path, args.model_path, args.fps, args.f1)
