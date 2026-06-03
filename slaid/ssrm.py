import math

import cv2
import numpy as np
from datasets.data_utils import get_bbox_from_mask, pad_to_square

from .ahfg import build_guidance_224
from .io import safe_resize_binary_wh


def bbox_from_mask_safe(mask):
    if mask is None or mask.sum() == 0:
        h, w = mask.shape[:2]
        return 0, h, 0, w
    return get_bbox_from_mask(mask.astype(np.uint8))
def clip_box(y1, y2, x1, x2, H, W):
    y1 = max(0, min(H - 1, int(round(y1))))
    y2 = max(y1 + 1, min(H, int(round(y2))))
    x1 = max(0, min(W - 1, int(round(x1))))
    x2 = max(x1 + 1, min(W, int(round(x2))))
    return y1, y2, x1, x2
def estimate_orientation(mask):
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
def major_is_horizontal(angle):
    return abs(math.cos(angle)) >= abs(math.sin(angle))
def defect_shape_stats(mask):
    y1, y2, x1, x2 = bbox_from_mask_safe(mask)
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    H, W = mask.shape[:2]
    angle = estimate_orientation(mask)
    horiz = major_is_horizontal(angle)
    major = w if horiz else h
    minor = h if horiz else w
    span = major / float(W if horiz else H)
    aspect = major / float(max(1, minor))
    return {
        'bbox': (y1, y2, x1, x2),
        'height': h,
        'width': w,
        'angle': angle,
        'horizontal': horiz,
        'major_extent': major,
        'minor_extent': minor,
        'span_ratio': span,
        'aspect_ratio': aspect,
    }
def choose_branch(mask, span_ratio_thresh=0.93, aspect_ratio_thresh=12.0):
    stats = defect_shape_stats(mask)
    if stats['span_ratio'] >= span_ratio_thresh and stats['aspect_ratio'] >= aspect_ratio_thresh:
        return 'sliding', stats
    return 'local', stats
def _sigmoid(x):
    x = float(np.clip(x, -50.0, 50.0))
    return 1.0 / (1.0 + math.exp(-x))
def compute_detail_strength(stats):
    aspect = float(stats.get('aspect_ratio', 1.0))
    span = float(stats.get('span_ratio', 0.0))
    minor = float(stats.get('minor_extent', 1.0))

    elongation_gate = _sigmoid((aspect - 4.0) / 1.8)
    span_gate = _sigmoid((span - 0.34) / 0.16)
    thin_gate = _sigmoid((10.0 - minor) / 3.0)

    alpha = elongation_gate * (0.35 + 0.45 * span_gate + 0.20 * thin_gate)
    if aspect < 3.0:
        alpha *= 0.25
    return float(np.clip(alpha, 0.0, 1.0))
def choose_detail_window_count(stats, max_windows=3, span_ratio_thresh=0.93, aspect_ratio_thresh=12.0):
    alpha = compute_detail_strength(stats)
    aspect = float(stats.get('aspect_ratio', 1.0))
    span = float(stats.get('span_ratio', 0.0))

    if alpha < 0.18:
        return 0, alpha
    if span >= span_ratio_thresh and aspect >= aspect_ratio_thresh:
        return max(1, int(max_windows)), alpha
    if span >= 0.68 and aspect >= 8.0:
        return max(2, min(int(max_windows), 4)), alpha
    if span >= 0.45 and aspect >= 5.5:
        return max(2, min(int(max_windows), 3)), alpha
    return 1, alpha
def make_dilated_context(mask, minor_extent, major_extent, extra_scale=1.0):
    radius = int(max(8, minor_extent * (1.1 * extra_scale), major_extent * 0.015 * extra_scale))
    radius = min(radius, 56)
    k = max(3, radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    context = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return context.astype(np.uint8)
def expand_axis_aligned_box(image_shape, mask, angle, along_scale=1.8, across_scale=1.6, min_size=32):
    H, W = image_shape[:2]
    y1, y2, x1, x2 = bbox_from_mask_safe(mask)
    cy = 0.5 * (y1 + y2)
    cx = 0.5 * (x1 + x2)
    box_h = max(min_size, y2 - y1)
    box_w = max(min_size, x2 - x1)

    if major_is_horizontal(angle):
        half_h = max(min_size / 2.0, box_h * across_scale / 2.0)
        half_w = max(min_size / 2.0, box_w * along_scale / 2.0)
    else:
        half_h = max(min_size / 2.0, box_h * along_scale / 2.0)
        half_w = max(min_size / 2.0, box_w * across_scale / 2.0)

    return clip_box(cy - half_h, cy + half_h, cx - half_w, cx + half_w, H, W)
def crop_by_box(image, box):
    y1, y2, x1, x2 = box
    return image[y1:y2, x1:x2].copy()
def crop_masks_by_box(mask_a, mask_b, box):
    y1, y2, x1, x2 = box
    return mask_a[y1:y2, x1:x2].copy(), mask_b[y1:y2, x1:x2].copy()
def build_masked_reference(crop_image, context_mask, fill_mode='mean'):
    ctx = context_mask.astype(bool)
    masked = crop_image.copy().astype(np.uint8)
    if fill_mode == 'mean' and ctx.any():
        mean_color = crop_image[ctx].reshape(-1, 3).mean(axis=0)
        fill_color = np.round(mean_color).astype(np.uint8)
    else:
        fill_color = np.array([255, 255, 255], dtype=np.uint8)
    masked[~ctx] = fill_color
    return masked
def zoom_image_and_masks(image, context_mask, defect_mask, target_coverage=0.80, max_zoom=2.6):
    stats = defect_shape_stats(defect_mask)
    crop_h, crop_w = image.shape[:2]
    crop_major = crop_w if stats['horizontal'] else crop_h
    current_coverage = stats['major_extent'] / float(max(1, crop_major))
    thin_bonus = 1.0
    if stats['minor_extent'] <= 4:
        thin_bonus = 1.35
    elif stats['minor_extent'] <= 8:
        thin_bonus = 1.18
    zoom = np.clip((target_coverage / max(current_coverage, 1e-6)) * thin_bonus, 1.0, max_zoom)

    if zoom <= 1.01:
        return image, context_mask, defect_mask, 1.0

    new_w = max(8, int(round(image.shape[1] * zoom)))
    new_h = max(8, int(round(image.shape[0] * zoom)))

    image_big = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ctx_big = safe_resize_binary_wh(context_mask, (new_w, new_h))
    def_big = safe_resize_binary_wh(defect_mask, (new_w, new_h))

    cy = new_h // 2
    cx = new_w // 2
    half_h = crop_h // 2
    half_w = crop_w // 2

    y1 = max(0, cy - half_h)
    x1 = max(0, cx - half_w)
    y2 = min(new_h, y1 + crop_h)
    x2 = min(new_w, x1 + crop_w)

    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)

    image_zoom = image_big[y1:y2, x1:x2].copy()
    ctx_zoom = ctx_big[y1:y2, x1:x2].copy()
    def_zoom = def_big[y1:y2, x1:x2].copy()
    return image_zoom, ctx_zoom, def_zoom, float(zoom)
def pad_to_square_two_masks(image, context_mask, defect_mask, pad_value_mode='mean'):
    H, W = image.shape[:2]

    if context_mask.astype(bool).any():
        fill_color = np.round(image[context_mask.astype(bool)].reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    else:
        if pad_value_mode == 'white':
            fill_color = np.array([255, 255, 255], dtype=np.uint8)
        else:
            fill_color = np.round(image.reshape(-1, 3).mean(axis=0)).astype(np.uint8)

    if H == W:
        return image, context_mask.astype(np.uint8), defect_mask.astype(np.uint8)

    if H > W:
        pad1 = (H - W) // 2
        pad2 = H - W - pad1
        image = cv2.copyMakeBorder(image, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=fill_color.tolist())
        context_mask = cv2.copyMakeBorder(context_mask.astype(np.uint8), 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=0)
        defect_mask = cv2.copyMakeBorder(defect_mask.astype(np.uint8), 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=0)
    else:
        pad1 = (W - H) // 2
        pad2 = W - H - pad1
        image = cv2.copyMakeBorder(image, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=fill_color.tolist())
        context_mask = cv2.copyMakeBorder(context_mask.astype(np.uint8), pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=0)
        defect_mask = cv2.copyMakeBorder(defect_mask.astype(np.uint8), pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=0)

    return image, context_mask, defect_mask
def to_ref_224(image, context_mask, defect_mask):
    image_sq, context_sq, defect_sq = pad_to_square_two_masks(
        image.astype(np.uint8),
        context_mask.astype(np.uint8),
        defect_mask.astype(np.uint8),
        pad_value_mode='mean'
    )
    image_224 = cv2.resize(image_sq, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    context_224 = safe_resize_binary_wh(context_sq, (224, 224)).astype(np.uint8)
    defect_224 = safe_resize_binary_wh(defect_sq, (224, 224)).astype(np.uint8)
    return image_224, context_224, defect_224
def build_sliding_windows(crop_image, crop_context, crop_defect, angle, max_windows=3, overlap=0.55):
    H, W = crop_image.shape[:2]
    horiz = major_is_horizontal(angle)
    stats = defect_shape_stats(crop_defect)
    major_extent = stats['major_extent']
    minor_extent = stats['minor_extent']

    total_major = W if horiz else H
    win_major = int(max(80, min(total_major, max(major_extent * 0.28, minor_extent * 12.0))))
    win_major = min(total_major, max(48, win_major))
    step = max(16, int(round(win_major * (1.0 - overlap))))

    if total_major <= win_major:
        starts = [0]
    else:
        starts = list(range(0, max(1, total_major - win_major + 1), step))
        if starts[-1] != total_major - win_major:
            starts.append(total_major - win_major)

    candidates = []
    for s in starts:
        e = min(total_major, s + win_major)
        if horiz:
            win_img = crop_image[:, s:e, :]
            win_ctx = crop_context[:, s:e]
            win_def = crop_defect[:, s:e]
        else:
            win_img = crop_image[s:e, :, :]
            win_ctx = crop_context[s:e, :]
            win_def = crop_defect[s:e, :]

        score = int(win_def.sum())
        if score <= 0:
            continue
        candidates.append({
            'start': int(s),
            'end': int(e),
            'score': score,
            'center': float(s + e) / 2.0,
            'image': win_img.copy(),
            'context': win_ctx.copy(),
            'defect': win_def.copy(),
        })

    if len(candidates) == 0:
        candidates = [{
            'start': 0,
            'end': total_major,
            'score': int(crop_defect.sum()),
            'center': float(total_major) / 2.0,
            'image': crop_image.copy(),
            'context': crop_context.copy(),
            'defect': crop_defect.copy(),
        }]

    selected = []
    min_gap = max(1.0, win_major * 0.38)
    for cand in sorted(candidates, key=lambda x: x['score'], reverse=True):
        if all(abs(cand['center'] - s['center']) >= min_gap for s in selected):
            selected.append(cand)
        if len(selected) >= max_windows:
            break

    selected = sorted(selected, key=lambda x: x['center'])
    return selected
def prepare_window_ref(win_img, win_ctx, win_def, target_coverage=0.82, max_zoom=2.8):
    masked = build_masked_reference(win_img, win_ctx, fill_mode='mean')
    zoom_img, zoom_ctx, zoom_def, zoom_factor = zoom_image_and_masks(
        masked, win_ctx, win_def, target_coverage=target_coverage, max_zoom=max_zoom
    )
    ref_224, ctx_224, def_224 = to_ref_224(zoom_img, zoom_ctx, zoom_def)
    guidance_224, hf_224 = build_guidance_224(ref_224, ctx_224, def_224)
    return {
        'ref_224': ref_224,
        'ctx_224': ctx_224,
        'def_224': def_224,
        'guidance_224': guidance_224,
        'hf_224': hf_224,
        'zoom_factor': zoom_factor,
    }
def build_window_meta(windows, total_major):
    total_major = float(max(1.0, total_major))
    meta = []
    for win in windows:
        start = float(win['start']) / total_major
        end = float(win['end']) / total_major
        center = 0.5 * (start + end)
        span = max(1.0, float(win['end'] - win['start'])) / total_major
        meta.append([start, end, center, span])
    return np.array(meta, dtype=np.float32)
def build_window_reliability(windows):
    if len(windows) == 0:
        return np.zeros((0,), dtype=np.float32)

    scores = np.array([max(1.0, float(w.get('score', 1.0))) for w in windows], dtype=np.float32)
    score_norm = scores / max(1e-6, float(scores.max()))
    reliability = []
    for idx, win in enumerate(windows):
        def_mask = win.get('def_224')
        hf = win.get('hf_224')
        if def_mask is not None and np.any(def_mask > 0) and hf is not None:
            hf_val = cv2.cvtColor(hf.astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32) / 255.0
            hf_score = float(hf_val[def_mask > 0].mean()) if np.any(def_mask > 0) else 0.0
        else:
            hf_score = 0.0
        zoom = float(win.get('zoom_factor', 1.0))
        zoom_gate = 1.0 if zoom <= 2.6 else max(0.55, 1.0 - 0.12 * (zoom - 2.6))
        rel = (0.35 + 0.45 * float(score_norm[idx]) + 0.20 * hf_score) * zoom_gate
        reliability.append(float(np.clip(rel, 0.20, 1.0)))
    return np.array(reliability, dtype=np.float32)
def build_reference_windows(ref_image, ref_mask, max_windows=3, span_ratio_thresh=0.93, aspect_ratio_thresh=12.0):
    _, stats = choose_branch(ref_mask, span_ratio_thresh=span_ratio_thresh, aspect_ratio_thresh=aspect_ratio_thresh)
    context_mask = make_dilated_context(ref_mask, stats['minor_extent'], stats['major_extent'], extra_scale=1.0)
    detail_count, alpha_detail = choose_detail_window_count(
        stats,
        max_windows=max_windows,
        span_ratio_thresh=span_ratio_thresh,
        aspect_ratio_thresh=aspect_ratio_thresh,
    )

    global_box = expand_axis_aligned_box(
        ref_image.shape[:2],
        context_mask,
        stats['angle'],
        along_scale=1.85,
        across_scale=2.65,
        min_size=96,
    )
    global_image = crop_by_box(ref_image, global_box)
    global_ctx, global_def = crop_masks_by_box(context_mask, ref_mask, global_box)
    global_view = prepare_window_ref(
        global_image,
        global_ctx,
        global_def,
        target_coverage=0.50,
        max_zoom=1.35,
    )
    global_view['score'] = int(global_def.sum())
    global_view['start'] = 0
    global_view['end'] = max(global_image.shape[:2])

    if detail_count <= 0:
        mode = 'global'
        windows = []
        total_major = global_image.shape[1] if stats['horizontal'] else global_image.shape[0]
    elif detail_count == 1:
        mode = 'local'
        crop_box = expand_axis_aligned_box(ref_image.shape[:2], context_mask, stats['angle'], along_scale=1.55, across_scale=2.0, min_size=56)
        crop_image = crop_by_box(ref_image, crop_box)
        crop_ctx, crop_def = crop_masks_by_box(context_mask, ref_mask, crop_box)
        win = prepare_window_ref(crop_image, crop_ctx, crop_def, target_coverage=0.82, max_zoom=2.8)
        win['score'] = int(crop_def.sum())
        win['start'] = 0
        win['end'] = max(crop_image.shape[:2])
        windows = [win]
        total_major = crop_image.shape[1] if stats['horizontal'] else crop_image.shape[0]
    else:
        mode = 'sliding'
        crop_box = expand_axis_aligned_box(ref_image.shape[:2], context_mask, stats['angle'], along_scale=1.05, across_scale=2.25, min_size=72)
        crop_image = crop_by_box(ref_image, crop_box)
        crop_ctx, crop_def = crop_masks_by_box(context_mask, ref_mask, crop_box)
        raw_windows = build_sliding_windows(crop_image, crop_ctx, crop_def, stats['angle'], max_windows=detail_count, overlap=0.55)
        windows = []
        for raw in raw_windows:
            win = prepare_window_ref(raw['image'], raw['context'], raw['defect'], target_coverage=0.82, max_zoom=2.8)
            win['score'] = int(raw['score'])
            win['start'] = int(raw['start'])
            win['end'] = int(raw['end'])
            windows.append(win)
        total_major = crop_image.shape[1] if stats['horizontal'] else crop_image.shape[0]

    if len(windows) > 0:
        scores = np.array([max(1, int(w['score'])) for w in windows], dtype=np.float32)
        weights = scores / max(1e-6, scores.sum())
        rep_idx = int(np.argmax(scores))
        rep = windows[rep_idx]
        window_meta = build_window_meta(windows, total_major)
        reliability = build_window_reliability(windows)
        debug_strip = cv2.hconcat([cv2.resize(w['ref_224'], (112, 224)) for w in windows])
    else:
        weights = np.zeros((0,), dtype=np.float32)
        rep = global_view
        window_meta = np.zeros((0, 4), dtype=np.float32)
        reliability = np.zeros((0,), dtype=np.float32)
        debug_strip = global_view['ref_224'].copy()

    return {
        'mode': mode,
        'stats': stats,
        'windows': windows,
        'weights': weights.astype(np.float32),
        'window_meta': window_meta,
        'window_reliability': reliability.astype(np.float32),
        'alpha_detail': float(alpha_detail if len(windows) > 0 else 0.0),
        'global_ref_224': global_view['ref_224'].copy(),
        'global_ctx_224': global_view['ctx_224'].copy(),
        'global_hf_224': global_view['hf_224'].copy(),
        'rep_ref_224': rep['ref_224'].copy(),
        'rep_ctx_224': rep['ctx_224'].copy(),
        'rep_def_224': rep['def_224'].copy(),
        'rep_guidance_224': rep['guidance_224'].copy(),
        'rep_hf_224': rep['hf_224'].copy(),
        'debug_strip': debug_strip,
    }
def build_target_crop_box(tar_image, tar_mask, ref_stats):
    angle = ref_stats['angle']
    if ref_stats['span_ratio'] >= 0.93 and ref_stats['aspect_ratio'] >= 12.0:
        along_scale = 1.12
        across_scale = 1.85
    else:
        along_scale = 1.45
        across_scale = 1.55
    return expand_axis_aligned_box(tar_image.shape[:2], tar_mask, angle, along_scale=along_scale, across_scale=across_scale, min_size=72)
def build_target_prior(tar_mask_crop, ref_stats):
    extra_scale = 1.10 if ref_stats['minor_extent'] <= 6 else 1.0
    prior = make_dilated_context(tar_mask_crop, ref_stats['minor_extent'], ref_stats['major_extent'], extra_scale=extra_scale)
    k = max(5, int(round(max(5, ref_stats['minor_extent'] * 2.5))))
    if k % 2 == 0:
        k += 1
    prior = cv2.GaussianBlur(prior.astype(np.float32), (k, k), 0)
    if prior.max() > 1e-6:
        prior = prior / prior.max()
    return prior.astype(np.float32)
def soft_normalize_map(x, eps=1e-6):
    x = x.astype(np.float32)
    mx = float(x.max())
    if mx <= eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / mx, 0.0, 1.0).astype(np.float32)
def resize_soft_map_to_square(map_2d, out_size=512):
    square = pad_to_square(map_2d[:, :, None].astype(np.float32), pad_value=0, random=False)[:, :, 0]
    return cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
def build_target_segment_maps(tar_mask_crop, ref_pack, prior_soft, out_size=512):
    windows = ref_pack['windows']
    num_windows = len(windows)
    region = soft_normalize_map(prior_soft)

    if tar_mask_crop.sum() > 0:
        edge = cv2.morphologyEx(tar_mask_crop.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
        edge = cv2.GaussianBlur(edge, (5, 5), 0)
        hf_support = np.maximum(edge, tar_mask_crop.astype(np.float32))
        hf_support = soft_normalize_map(hf_support)
    else:
        hf_support = region.copy()

    if num_windows <= 0:
        assignments = np.zeros((0, out_size, out_size), dtype=np.float32)
        return {
            'region': resize_soft_map_to_square(region, out_size),
            'detail': resize_soft_map_to_square(hf_support, out_size),
            'assignments': assignments,
        }

    stats = defect_shape_stats(tar_mask_crop)
    H, W = tar_mask_crop.shape[:2]
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    mask_bool = tar_mask_crop > 0

    if mask_bool.sum() >= 2:
        cx = float(xx[mask_bool].mean())
        cy = float(yy[mask_bool].mean())
    else:
        cx = W * 0.5
        cy = H * 0.5

    angle = float(stats['angle'])
    coord = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
    support = np.maximum(region, tar_mask_crop.astype(np.float32))
    support_bool = support > 0.03
    vals = coord[support_bool]
    if vals.size == 0:
        vals = coord.reshape(-1)
    cmin, cmax = float(vals.min()), float(vals.max())
    if cmax - cmin < 1e-6:
        u = np.zeros_like(coord, dtype=np.float32) + 0.5
    else:
        u = ((coord - cmin) / (cmax - cmin)).astype(np.float32)
    u = np.clip(u, 0.0, 1.0)

    meta = ref_pack['window_meta']
    if meta.shape[0] != num_windows:
        centers = np.linspace(0.0, 1.0, num_windows, dtype=np.float32)
        spans = np.ones(num_windows, dtype=np.float32) / max(1, num_windows)
    else:
        centers = meta[:, 2].astype(np.float32)
        spans = np.clip(meta[:, 3].astype(np.float32), 1.0 / max(1, num_windows), 1.0)

    raw = []
    for center, span in zip(centers, spans):
        sigma = max(0.08, float(span) * 0.38)
        weight = np.exp(-0.5 * ((u - float(center)) / sigma) ** 2).astype(np.float32)
        weight *= region
        raw.append(weight)
    raw = np.stack(raw, axis=0)

    denom = raw.sum(axis=0, keepdims=True)
    if float(denom.max()) <= 1e-6:
        raw[:] = region[None, :, :] / max(1, num_windows)
        denom = raw.sum(axis=0, keepdims=True)
    assignments = raw / np.maximum(denom, 1e-6)

    assignments_512 = []
    for i in range(num_windows):
        assignments_512.append(resize_soft_map_to_square(assignments[i], out_size))
    assignments_512 = np.stack(assignments_512, axis=0).astype(np.float32)

    return {
        'region': resize_soft_map_to_square(region, out_size),
        'detail': resize_soft_map_to_square(hf_support, out_size),
        'assignments': assignments_512,
    }
