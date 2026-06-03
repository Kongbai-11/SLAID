import cv2
import numpy as np


def _morph_skeleton(binary_mask):
    img = (binary_mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _estimate_orientation(mask):
    pts = np.column_stack(np.where(mask > 0))
    if len(pts) < 2:
        return 0.0
    ys = pts[:, 0].astype(np.float32)
    xs = pts[:, 1].astype(np.float32)
    coords = np.stack([xs, ys], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    if np.any(~np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    return float(np.arctan2(v[1], v[0]))


def _major_is_horizontal(angle):
    return abs(np.cos(angle)) >= abs(np.sin(angle))


def _fft_periodic_residual(gray_float, top_k=8, center_radius=7, notch_radius=3, attenuate=0.10):
    h, w = gray_float.shape
    F = np.fft.fftshift(np.fft.fft2(gray_float))
    mag = np.log1p(np.abs(F))
    cy, cx = h // 2, w // 2

    yy, xx = np.ogrid[:h, :w]
    central = (yy - cy) ** 2 + (xx - cx) ** 2 <= center_radius ** 2
    mag_work = mag.copy()
    mag_work[central] = 0.0

    notch_mask = np.ones((h, w), dtype=np.float32)
    flat_idx = np.argpartition(mag_work.ravel(), -min(top_k, mag_work.size))[-min(top_k, mag_work.size):]
    pts = np.column_stack(np.unravel_index(flat_idx, mag_work.shape))

    for py, px in pts:
        if (py - cy) ** 2 + (px - cx) ** 2 <= center_radius ** 2:
            continue
        sy, sx = 2 * cy - py, 2 * cx - px
        cv2.circle(notch_mask, (int(px), int(py)), int(notch_radius), float(attenuate), -1)
        if 0 <= sy < h and 0 <= sx < w:
            cv2.circle(notch_mask, (int(sx), int(sy)), int(notch_radius), float(attenuate), -1)

    F_filtered = F * notch_mask
    recon = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered))).astype(np.float32)
    residual = np.abs(gray_float - recon)
    return residual
def _normalize_in_support(x, support, p_lo=40.0, p_hi=99.5):
    vals = x[support > 0]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
def dog(img, mask, thresh=18, top_k=8, notch_radius=3, keep_percent=45.0):
    """
    FFT + mask-geometry high-frequency map.
    Output style is close to official AnyDoor high-frequency guidance:
    black background + sparse colored lines in 3-channel uint8.
    """
    H, W = img.shape[:2]
    img_256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    mask_256 = (cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)

    if mask_256.sum() == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    mask_erode = cv2.erode(mask_256, np.ones((5, 5), np.uint8), iterations=1)
    if mask_erode.sum() < 16:
        mask_erode = mask_256.copy()

    angle = _estimate_orientation(mask_erode)
    vertical = not _major_is_horizontal(angle)
    band_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15) if vertical else (15, 5))
    support = cv2.dilate(mask_erode, band_kernel, iterations=1)

    gray = cv2.cvtColor(img_256, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    residual = _fft_periodic_residual(gray, top_k=top_k, center_radius=7, notch_radius=notch_radius, attenuate=0.10)
    residual_n = _normalize_in_support(residual, support, p_lo=30.0, p_hi=99.7)

    gradx = cv2.Sobel((gray * 255).astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel((gray * 255).astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
    grad = 0.5 * np.abs(gradx) + 0.5 * np.abs(grady)
    grad_n = _normalize_in_support(grad.astype(np.float32), support, p_lo=55.0, p_hi=99.5)

    edge = cv2.morphologyEx(mask_erode, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    edge = np.clip(edge, 0.0, 1.0)

    center = _morph_skeleton(mask_erode).astype(np.float32)
    if center.sum() == 0:
        center = mask_erode.astype(np.float32)
    center = cv2.dilate(center, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    center = np.clip(center, 0.0, 1.0)

    resp = 0.42 * residual_n + 0.12 * grad_n + 0.24 * edge + 0.22 * center
    resp *= support.astype(np.float32)

    inside_vals = resp[mask_erode > 0]
    if inside_vals.size == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    keep_thr = float(np.percentile(inside_vals, max(0.0, 100.0 - float(keep_percent))))
    base_thr = max(thresh / 255.0, keep_thr)
    binary = (resp >= base_thr).astype(np.uint8)
    binary = np.maximum(binary, center.astype(np.uint8))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7) if vertical else (7, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    binary = (binary * support).astype(np.uint8)

    intensity = np.zeros_like(resp, dtype=np.float32)
    intensity += binary.astype(np.float32) * 0.72
    intensity += np.clip(resp, 0.0, 1.0) * 0.28
    intensity = np.clip(intensity, 0.0, 1.0)
    intensity = cv2.GaussianBlur(intensity, (3, 3), 0)
    intensity = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)

    # Keep the existing HF extraction unchanged and only colorize the final map
    # with the source hue/saturation while reusing HF intensity as value.
    src_hsv = cv2.cvtColor(img_256.astype(np.uint8), cv2.COLOR_RGB2HSV)
    hf_hsv = src_hsv.copy()
    hf_hsv[:, :, 2] = intensity
    hf = cv2.cvtColor(hf_hsv, cv2.COLOR_HSV2RGB)
    hf = cv2.resize(hf, (W, H), interpolation=cv2.INTER_LINEAR)
    return hf
def build_guidance_224(ref_224, ctx_224, def_224):
    hf_map = dog(
        ref_224.astype(np.uint8),
        def_224.astype(np.uint8),
        thresh=18,
        top_k=8,
        notch_radius=3,
        keep_percent=45.0,
    )

    # Keep the representative guidance visually close to the reference
    # and let hf_map carry the official-style sparse signal.
    guidance = ref_224.astype(np.uint8).copy()
    return guidance, hf_map.astype(np.uint8)
def color_match_lab_l_only_rgb(src_rgb, ref_rgb, mask=None, eps=1e-6):
    src_lab = cv2.cvtColor(src_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    if mask is None:
        mask_bool = np.ones(src_lab.shape[:2], dtype=bool)
    else:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_bool = mask > 0.05
        if mask_bool.sum() < 16:
            mask_bool = np.ones(src_lab.shape[:2], dtype=bool)

    out = src_lab.copy()
    s = src_lab[:, :, 0][mask_bool]
    r = ref_lab[:, :, 0][mask_bool]
    s_mean, s_std = float(s.mean()), float(s.std())
    r_mean, r_std = float(r.mean()), float(r.std())
    out[:, :, 0] = (src_lab[:, :, 0] - s_mean) * (r_std / (s_std + eps)) + r_mean

    # Keep chroma from the prediction to avoid washing out the thin defect.
    out[:, :, 1] = src_lab[:, :, 1]
    out[:, :, 2] = src_lab[:, :, 2]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
def build_blend_alpha(mask, ksize=21, core=0.92, edge=0.03):
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0

    soft = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    if soft.max() > 1e-6:
        soft = soft / soft.max()

    alpha = edge + (core - edge) * soft
    return np.clip(alpha, 0.0, 1.0)[:, :, None]
