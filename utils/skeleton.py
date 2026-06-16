import numpy as np
from skimage.morphology import skeletonize


def extract_wireframe(binary_image: np.ndarray) -> np.ndarray:
    """Return 1-pixel structural skeleton with black ink on white background."""
    ink_mask = binary_image == 0
    skeleton_mask = skeletonize(ink_mask)
    return np.where(skeleton_mask, 0, 255).astype(np.uint8)
