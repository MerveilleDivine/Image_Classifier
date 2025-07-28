# ===== Colab Setup: Install dependencies =====
# Run this cell first to install all required libraries
!pip install --upgrade torch torchvision gradio opencv-python scikit-learn matplotlib

# ===== Imports =====
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
from matplotlib import pyplot as plt
plt.ion()  # Enable interactive mode
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from PIL import Image
import cv2
import gradio as gr
import torch.nn.functional as F  # For softmax probabilities
# ===== Constants =====
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# ===== 1. DIP Enhancement, Segmentation, Edges, Morphology =====
def enhance_image(cv_img):
    """
    Apply Gaussian blur and histogram equalization (contrast enhancement).
    """
    blur = cv2.GaussianBlur(cv_img, (5, 5), 0)
    yuv = cv2.cvtColor(blur, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def segment_image(cv_img):
    """Simple segmentation via Otsu's thresholding."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.merge([mask, mask, mask])

def edge_detect(cv_img, method='canny'):
    """Return edge map using Canny or Sobel."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    if method == 'sobel':
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        edges = np.uint8(np.clip(mag / mag.max() * 255, 0, 255))
    else:
        med = np.median(gray)
        lower = int(max(0, 0.7 * med))
        upper = int(min(255, 1.3 * med))
        edges = cv2.Canny(gray, lower, upper)
    return edges

def morph_cleanup(mask, use_open=True, kernel_size=3, iterations=1):
    """Apply opening (erosion then dilation) or closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if use_open:
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    else:
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

# Combined DIP pipeline for PIL images
def transform_pil(img):
    """Enhance, segment, and clean via morphology."""
    img_np = np.array(img)
    enhanced = enhance_image(img_np)
    mask = segment_image(enhanced)
    clean_mask = morph_cleanup(mask[:,:,0], use_open=True)
    clean_mask = cv2.merge([clean_mask]*3)
    segmented = cv2.bitwise_and(enhanced, clean_mask)
    return Image.fromarray(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))

# Compare original vs equalized contrast
import matplotlib.pyplot as plt
def compare_eq_vs_raw(pil_img):
    img_np = np.array(pil_img)
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    eq_rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    axs[0].imshow(pil_img); axs[0].set_title('Original'); axs[0].axis('off')
    axs[1].imshow(eq_rgb); axs[1].set_title('Histogram Equalized'); axs[1].axis('off')
    plt.tight_layout(); plt.show()