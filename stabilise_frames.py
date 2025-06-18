import cv2
import numpy as np
from typing import List


def stabilise_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Feature-based global motion compensation.

    Detect ORB features on consecutive frame pairs, estimate homography with
    RANSAC, warp frames so the background is steady and return the aligned
    list. The output has the same dtype and shape as the input.
    """
    if len(frames) < 2:
        return frames

    height, width = frames[0].shape[:2]
    orb = cv2.ORB_create(500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    transforms = [np.eye(3, dtype=np.float32)]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

    for idx in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
        k1, d1 = orb.detectAndCompute(prev_gray, None)
        k2, d2 = orb.detectAndCompute(curr_gray, None)
        if d1 is None or d2 is None:
            transforms.append(transforms[-1])
            prev_gray = curr_gray
            continue
        matches = matcher.match(d1, d2)
        if len(matches) < 4:
            transforms.append(transforms[-1])
            prev_gray = curr_gray
            continue
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        if H is None:
            H = transforms[-1]
        else:
            H = H.astype(np.float32)
        transforms.append(H @ transforms[-1])
        prev_gray = curr_gray

    stabilised = []
    for frame, H in zip(frames, transforms):
        warped = cv2.warpPerspective(frame, H, (width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        stabilised.append(warped)

    return stabilised


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stabilise video frames")
    parser.add_argument("video_path", help="Path to video to process")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for extraction")
    parser.add_argument("--plot", action="store_true", help="Save before/after comparison")
    args = parser.parse_args()

    from extract_frames import extract_frames

    frames = extract_frames(args.video_path, args.fps)
    if not frames:
        raise SystemExit("No frames extracted")

    aligned = stabilise_frames(frames)
    print(f"Stabilised {len(aligned)} frames")

    if args.plot:
        before_after = cv2.hconcat([frames[0], aligned[0]])
        out_path = "stabilise_demo.png"
        cv2.imwrite(out_path, cv2.cvtColor(before_after, cv2.COLOR_RGB2BGR))
        print(f"Saved comparison to {out_path}")
