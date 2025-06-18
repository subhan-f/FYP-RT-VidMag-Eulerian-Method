import cv2
import numpy as np
from typing import List, Tuple


def crop_roi(frames: List[np.ndarray], marker_hsv: Tuple[int, int, int] = (0, 255, 255), pad: int = 10) -> List[np.ndarray]:
    """Crop a region containing a red marker across frames.

    Parameters
    ----------
    frames : List[np.ndarray]
        Input RGB frames.
    marker_hsv : Tuple[int, int, int], optional
        HSV colour of the marker, by default bright red ``(0, 255, 255)``.
    pad : int, optional
        Padding in pixels around detected ROI, by default 10.

    Returns
    -------
    List[np.ndarray]
        Cropped frames. If the marker is not found, returns original frames.
    """
    if not frames:
        return frames

    height, width = frames[0].shape[:2]

    # accumulate bounding boxes
    boxes = []
    tol = np.array([10, 100, 100])
    hsv_target = np.array(marker_hsv)

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lower = np.maximum(hsv_target - tol, (0, 0, 0))
        upper = np.minimum(hsv_target + tol, (179, 255, 255))
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 0:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        print("[WARN] Marker not found; returning full frames")
        return frames

    box = np.array(boxes)
    x0, y0 = box[:, :2].min(axis=0)
    x1, y1 = box[:, 2:].max(axis=0)

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)

    cropped = [f[y0:y1, x0:x1].copy() for f in frames]
    return cropped


if __name__ == "__main__":
    # synthetic test: circle on random noise
    h, w = 120, 160
    frames = []
    center = (w // 2, h // 2)
    for _ in range(5):
        img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        cv2.circle(img, center, 15, (0, 0, 255), -1)  # red in BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    cropped = crop_roi(frames)
    print("Original shape:", frames[0].shape)
    print("Cropped shape:", cropped[0].shape)
