import cv2
import numpy as np
from typing import List

from laplacian_utils import build_laplacian_pyramid, reconstruct_from_laplacian
from temporal_filter import temporal_bandpass, amplify


def magnify_motion(frames: List[np.ndarray], fs: int, f_low: float, f_high: float,
                   alpha: float, levels: int = 5) -> List[np.ndarray]:
    """Magnify subtle motions in ``frames`` using Eulerian video magnification.

    Parameters
    ----------
    frames : List[np.ndarray]
        RGB uint8 frames.
    fs : int
        Frame rate in Hz.
    f_low : float
        Low cutoff frequency.
    f_high : float
        High cutoff frequency.
    alpha : float
        Linear amplification factor.
    levels : int, optional
        Number of pyramid levels, by default 5.

    Returns
    -------
    List[np.ndarray]
        Magnified frames as BGR uint8 images suitable for writing with cv2.
    """
    if not frames:
        return []

    # Build Laplacian pyramids for all frames
    pyramids = [build_laplacian_pyramid(f, levels) for f in frames]
    n_levels = len(pyramids[0])
    T = len(frames)

    # Process each level independently across time
    for lvl in range(n_levels):
        stack = np.stack([pyr[lvl] for pyr in pyramids], axis=0)
        filtered = temporal_bandpass(stack, fs, f_low, f_high)
        amplified = amplify(filtered, alpha)
        for t in range(T):
            pyramids[t][lvl] = pyramids[t][lvl] + amplified[t]

    # Reconstruct frames
    out_frames = [reconstruct_from_laplacian(pyr) for pyr in pyramids]
    # Convert RGB to BGR for OpenCV writing
    out_frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in out_frames]
    return out_frames_bgr


if __name__ == "__main__":
    import argparse
    from extract_frames import extract_frames

    parser = argparse.ArgumentParser(description="Apply Eulerian motion magnification")
    parser.add_argument("video_path", help="Input MP4")
    parser.add_argument("--fps", type=int, default=30, help="Resampled FPS")
    parser.add_argument("--flow", type=float, default=1.0, help="Low cut-off frequency")
    parser.add_argument("--fhigh", type=float, default=2.0, help="High cut-off frequency")
    parser.add_argument("--alpha", type=float, default=10.0, help="Amplification factor")
    parser.add_argument("--levels", type=int, default=5, help="Pyramid levels")
    args = parser.parse_args()

    frames = extract_frames(args.video_path, args.fps)
    if not frames:
        raise SystemExit("No frames extracted")

    magnified = magnify_motion(frames, args.fps, args.flow, args.fhigh, args.alpha, args.levels)

    out_path = args.video_path.rsplit(".", 1)[0] + "_magnified.mp4"
    height, width = magnified[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    for frame in magnified:
        writer.write(frame)
    writer.release()
    print(f"Wrote {out_path} with {len(magnified)} frames")
