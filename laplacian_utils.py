import cv2
import numpy as np
from typing import List


def build_laplacian_pyramid(img: np.ndarray, levels: int = 5) -> List[np.ndarray]:
    """Construct a Laplacian pyramid.

    Parameters
    ----------
    img : np.ndarray
        Input image (RGB uint8).
    levels : int, optional
        Number of levels, by default 5.

    Returns
    -------
    List[np.ndarray]
        Pyramid with ``levels`` Laplacian images and the final Gaussian level.
    """
    pyr = []
    current = img.astype(np.float32)
    for _ in range(levels):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        lap = current - up
        pyr.append(lap)
        current = down
    pyr.append(current)
    return pyr


def reconstruct_from_laplacian(pyr: List[np.ndarray]) -> np.ndarray:
    """Reconstruct an image from its Laplacian pyramid."""
    current = pyr[-1]
    for lap in reversed(pyr[:-1]):
        up = cv2.pyrUp(current, dstsize=(lap.shape[1], lap.shape[0]))
        current = up + lap
    current = np.clip(current, 0, 255)
    return current.astype(np.uint8)


if __name__ == "__main__":
    # simple round-trip test using a random image
    img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    pyr = build_laplacian_pyramid(img, levels=5)
    recon = reconstruct_from_laplacian(pyr)
    mse = np.mean((img.astype(np.float32) - recon.astype(np.float32)) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-8))
    print(f"PSNR: {psnr:.2f} dB")

