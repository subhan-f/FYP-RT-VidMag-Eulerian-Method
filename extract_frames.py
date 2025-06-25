import cv2
import numpy as np
from typing import List


def extract_frames(video_path: str, fps: int = 30) -> List[np.ndarray]:
    """Read an MP4, resample it to the target fps using cv2.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    fps : int, optional
        Target frames per second, by default 30.

    Returns
    -------
    List[np.ndarray]
        List of RGB uint8 frames shaped (H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frames = []
    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        success, frame = cap.read()
    cap.release()

    if not frames or abs(orig_fps - fps) < 1e-2:
        return frames

    n_frames_out = max(1, int(round(len(frames) * fps / orig_fps)))
    indices = np.linspace(0, len(frames) - 1, n_frames_out).astype(int)
    return [frames[i] for i in indices]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")

    args = parser.parse_args()

    out_frames = extract_frames(args.video_path, args.fps)
    if out_frames:
        h, w = out_frames[0].shape[:2]
    else:
        h = w = 0
    print(f"Extracted {len(out_frames)} frames of shape {h}x{w}")
