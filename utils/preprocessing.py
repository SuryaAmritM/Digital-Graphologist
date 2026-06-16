import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def adaptive_threshold_ink(image_bgr: np.ndarray) -> np.ndarray:
    """Return binary image with black ink on white background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Mild denoising improves adaptive threshold stability on phone photos.
    denoised = cv2.medianBlur(gray, 3)

    window_size = 25
    sauvola_threshold = threshold_sauvola(denoised, window_size=window_size, k=0.2)
    binary = (denoised < sauvola_threshold).astype(np.uint8) * 255

    # Remove thin horizontal notebook lines while keeping handwriting structure.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    ruled_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cleaned = cv2.subtract(binary, ruled_lines)

    # Normalize to pure black ink (0) over white background (255).
    return np.where(cleaned > 0, 0, 255).astype(np.uint8)
