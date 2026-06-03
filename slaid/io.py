import os

import cv2
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Cannot read mask: {path}')
    thr = 0 if int(mask.max()) <= 1 else 127
    return (mask > thr).astype(np.uint8)


def safe_resize_binary(mask, size_hw):
    h, w = size_hw
    return (cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def safe_resize_binary_wh(mask, size_wh):
    w, h = size_wh
    return (cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
