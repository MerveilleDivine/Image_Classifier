import cv2
import numpy as np
from PIL import Image


def enhance_image(cv_img):
    """Apply Gaussian blur and histogram equalization to a BGR image."""
    blurred = cv2.GaussianBlur(cv_img, (5, 5), 0)
    yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def segment_image(cv_img):
    """Segment a BGR image using Otsu thresholding."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.merge([mask, mask, mask])


def edge_detect(cv_img, method="canny"):
    """Return an edge map using Canny or Sobel."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    if method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        max_value = magnitude.max()
        if max_value == 0:
            return np.zeros_like(gray, dtype=np.uint8)
        return np.uint8(np.clip(magnitude / max_value * 255, 0, 255))

    median = np.median(gray)
    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))
    return cv2.Canny(gray, lower, upper)


def morph_cleanup(mask, use_open=True, kernel_size=3, iterations=1):
    """Apply morphological opening or closing to clean a mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    operation = cv2.MORPH_OPEN if use_open else cv2.MORPH_CLOSE
    return cv2.morphologyEx(mask, operation, kernel, iterations=iterations)


def transform_pil(img):
    """Apply the DIP enhancement and segmentation pipeline to a PIL image."""
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    enhanced = enhance_image(bgr)
    mask = segment_image(enhanced)
    clean_mask = morph_cleanup(mask[:, :, 0], use_open=True)
    clean_mask = cv2.merge([clean_mask] * 3)
    segmented = cv2.bitwise_and(enhanced, clean_mask)
    return Image.fromarray(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
