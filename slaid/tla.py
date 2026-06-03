"""
Training-free target-domain latent anchoring for local defect synthesis.

This module replaces the previous SCGS/RAGS sampling correction. It keeps the
DDIM denoising trajectory anchored to the target crop outside the editable
defect region, while leaving the defect core free for the reference/detail and
ControlNet guidance modules.
"""

import cv2
import numpy as np
import torch


def _odd_kernel(size):
    size = int(max(3, size))
    return size if size % 2 == 1 else size + 1


def _resize_soft_mask(mask, size_hw):
    h, w = size_hw
    resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _pad_to_square_mask(mask):
    h, w = mask.shape[:2]
    if h == w:
        return mask.astype(np.float32)

    pad = abs(h - w)
    pad1 = pad // 2
    pad2 = pad - pad1
    if h > w:
        return np.pad(mask.astype(np.float32), ((0, 0), (pad1, pad2)), constant_values=0.0)
    return np.pad(mask.astype(np.float32), ((pad1, pad2), (0, 0)), constant_values=0.0)


def prepare_tla_masks(
    tar_mask,
    tar_box_yyxx_crop,
    latent_size=64,
    core_dilate_px=13,
    ring_dilate_px=31,
    blur_px=17,
):
    """
    Build latent-space edit/background weights from the target defect mask.

    Returns two 2D maps:
    - edit_weight: 1 in the defect core, smoothly decays through the boundary.
    - anchor_weight: 1 outside the edit area, 0 in the defect core.
    """
    y1, y2, x1, x2 = [int(v) for v in tar_box_yyxx_crop]
    h_img, w_img = tar_mask.shape[:2]
    y1 = max(0, min(y1, h_img))
    y2 = max(y1 + 1, min(y2, h_img))
    x1 = max(0, min(x1, w_img))
    x2 = max(x1 + 1, min(x2, w_img))

    crop = (tar_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    if crop.size == 0 or crop.sum() == 0:
        edit = np.zeros((latent_size, latent_size), dtype=np.float32)
        return edit, 1.0 - edit

    core_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel(core_dilate_px), _odd_kernel(core_dilate_px)),
    )
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel(ring_dilate_px), _odd_kernel(ring_dilate_px)),
    )
    core = cv2.dilate(crop, core_kernel, iterations=1).astype(np.float32)
    ring = cv2.dilate(crop, ring_kernel, iterations=1).astype(np.float32)

    soft = cv2.GaussianBlur(ring, (_odd_kernel(blur_px), _odd_kernel(blur_px)), 0)
    if float(soft.max()) > 1e-6:
        soft = soft / float(soft.max())
    edit = np.maximum(core, soft * 0.82)
    edit = _resize_soft_mask(_pad_to_square_mask(edit), (latent_size, latent_size))

    anchor = 1.0 - edit
    anchor = cv2.GaussianBlur(anchor, (3, 3), 0)
    anchor = np.clip(anchor, 0.0, 1.0).astype(np.float32)
    return edit.astype(np.float32), anchor


def protect_tla_anchor(anchor_weight, protection_maps, protect_strength=0.92):
    """
    Suppress anchoring where earlier modules expect defect/detail generation.

    protection_maps can include target region/detail maps from module one/two.
    They are resized to latent space and treated as a soft no-anchor prior.
    """
    anchor = np.asarray(anchor_weight, dtype=np.float32)
    if protection_maps is None:
        return np.clip(anchor, 0.0, 1.0).astype(np.float32)

    protect = np.zeros_like(anchor, dtype=np.float32)
    for src in protection_maps:
        if src is None:
            continue
        src = np.asarray(src, dtype=np.float32)
        if src.ndim == 3:
            src = src[:, :, 0]
        if src.size == 0:
            continue
        src = cv2.resize(src, (anchor.shape[1], anchor.shape[0]), interpolation=cv2.INTER_LINEAR)
        src = np.clip(src, 0.0, 1.0)
        protect = np.maximum(protect, src)

    if float(protect.max()) > 1e-6:
        protect = cv2.GaussianBlur(protect, (5, 5), 0)
        protect = np.clip(protect, 0.0, 1.0)
    anchor = anchor * (1.0 - float(protect_strength) * protect)
    return np.clip(anchor, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def encode_target_latent(runtime_model, target_crop_512):
    """
    Encode the padded 512x512 target crop to the diffusion latent scale.
    target_crop_512 is expected in [-1, 1], HWC RGB.
    """
    x = torch.from_numpy(target_crop_512.copy()).float().to(runtime_model.device)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2).contiguous()
    posterior = runtime_model.encode_first_stage(x)
    return runtime_model.get_first_stage_encoding(posterior).detach()


def _target_at_alpha(target_latent, noise, alpha_value):
    alpha = torch.as_tensor(alpha_value, device=target_latent.device, dtype=target_latent.dtype).view(1, 1, 1, 1)
    return alpha.sqrt() * target_latent + (1.0 - alpha).sqrt() * noise


def build_tla_initial_noise(shape, sampler, target_latent, anchor_weight, device='cuda'):
    """
    Initialize x_T so the background starts on the target noisy trajectory.
    """
    anchor_noise = torch.randn(shape, device=device)
    anchor = torch.from_numpy(anchor_weight.astype(np.float32)).to(device).view(1, 1, *anchor_weight.shape)
    anchor = torch.nn.functional.interpolate(anchor, size=shape[-2:], mode='bilinear', align_corners=False)
    anchor = torch.clamp(anchor, 0.0, 1.0).to(dtype=anchor_noise.dtype)
    anchor = anchor.repeat(shape[0], shape[1], 1, 1)

    target = target_latent.to(device=device, dtype=anchor_noise.dtype)
    target_t = _target_at_alpha(target, anchor_noise, sampler.ddim_alphas[-1])
    x_t = anchor_noise * (1.0 - anchor) + target_t * anchor
    return x_t, anchor_noise


def _anchor_strength(i, total_steps, base_strength):
    progress = float(i) / float(max(total_steps - 1, 1))
    if progress < 0.10:
        return float(base_strength)
    if progress < 0.58:
        return float(base_strength * (0.92 - 0.30 * (progress - 0.10) / 0.48))
    if progress < 0.82:
        return float(base_strength * (0.45 * (1.0 - (progress - 0.58) / 0.24)))
    return 0.0


class TargetLatentAnchor:
    """
    Callable DDIM correction that anchors non-edit latent regions to target x_t.
    """

    def __init__(
        self,
        sampler,
        target_latent,
        anchor_noise,
        anchor_weight,
        strength=0.82,
    ):
        self.sampler = sampler
        self.target_latent = target_latent.detach()
        self.anchor_noise = anchor_noise.detach()
        self.anchor_weight = np.asarray(anchor_weight, dtype=np.float32)
        self.strength = float(np.clip(strength, 0.0, 1.0))

    @torch.no_grad()
    def __call__(self, img, pred_x0, i, total_steps):
        del pred_x0
        strength = _anchor_strength(i, total_steps, self.strength)
        if strength <= 1e-6:
            return img

        device = img.device
        b, c, h, w = img.shape
        anchor = torch.from_numpy(self.anchor_weight).to(device=device, dtype=img.dtype).view(1, 1, *self.anchor_weight.shape)
        anchor = torch.nn.functional.interpolate(anchor, size=(h, w), mode='bilinear', align_corners=False)
        anchor = torch.clamp(anchor * strength, 0.0, 1.0).repeat(b, c, 1, 1)

        target = self.target_latent.to(device=device, dtype=img.dtype)
        if target.shape[0] != b:
            target = target.repeat(b, 1, 1, 1)
        noise = self.anchor_noise.to(device=device, dtype=img.dtype)
        if noise.shape[0] != b:
            noise = noise.repeat(b, 1, 1, 1)

        index = max(0, min(int(total_steps - i - 1), len(self.sampler.ddim_alphas_prev) - 1))
        target_t = _target_at_alpha(target, noise, self.sampler.ddim_alphas_prev[index])
        return img * (1.0 - anchor) + target_t * anchor
