import os
import cv2
import einops
import numpy as np
import torch
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import *
from omegaconf import OmegaConf
from .tla import (
    TargetLatentAnchor,
    build_tla_initial_noise,
    encode_target_latent,
    prepare_tla_masks,
    protect_tla_anchor,
)
from .io import (
    ensure_dir,
    read_mask,
    read_rgb,
)
from .ssrm import (
    bbox_from_mask_safe,
    build_reference_windows,
    build_target_crop_box,
    build_target_prior,
    build_target_segment_maps,
)
from .ahfg import (
    build_blend_alpha,
    color_match_lab_l_only_rgb,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

DEFAULT_INFERENCE_CONFIG = './configs/inference.yaml'
RUNTIME_INFERENCE_CONFIG = DEFAULT_INFERENCE_CONFIG
RUNTIME_PRETRAINED_MODEL = ''
RUNTIME_DDIM_STEPS = 50

model = None
ddim_sampler = None
model_signature = None


def load_inference_model(config_path=None, pretrained_model=None):
    global model, ddim_sampler, model_signature
    if config_path is None:
        config_path = RUNTIME_INFERENCE_CONFIG
    if pretrained_model is None:
        pretrained_model = RUNTIME_PRETRAINED_MODEL

    signature = (
        os.path.abspath(config_path),
        str(pretrained_model or '').strip(),
    )
    if model is not None and ddim_sampler is not None and model_signature == signature:
        return model, ddim_sampler

    config = OmegaConf.load(config_path)
    model_ckpt = str(pretrained_model or config.pretrained_model)
    model_config = config.config_file

    runtime_model = create_model(model_config).cpu()

    missing, unexpected = runtime_model.load_state_dict(
        load_state_dict(model_ckpt, location='cuda'),
        strict=False,
    )
    if missing:
        print(f'Missing keys while loading base checkpoint: {len(missing)}')
    if unexpected:
        print(f'Unexpected keys while loading base checkpoint: {len(unexpected)}')

    model = runtime_model.cuda()
    ddim_sampler = DDIMSampler(model)
    model_signature = signature
    return model, ddim_sampler


def compose_control_hint_from_target_crop(cropped_target_image, tar_mask_crop, ref_pack):
    rep_guidance = ref_pack['rep_guidance_224']
    rep_hf = ref_pack['rep_hf_224']
    prior_soft = build_target_prior(tar_mask_crop, ref_pack['stats'])
    y1, y2, x1, x2 = bbox_from_mask_safe(prior_soft > 0.05)
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)

    guide_patch = cv2.resize(rep_guidance.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    guide_gray = cv2.cvtColor(guide_patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    guide_gray_3 = np.stack([guide_gray, guide_gray, guide_gray], axis=-1)

    hf_patch = cv2.resize(rep_hf.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    base_patch = cropped_target_image[y1:y2, x1:x2, :].astype(np.float32)

    prior_box = prior_soft[y1:y2, x1:x2].astype(np.float32)
    prior_box_3 = np.stack([prior_box, prior_box, prior_box], axis=-1)

    # Recover the original HF strength from the value channel so the downstream
    # control logic stays close to the previous grayscale implementation.
    hf_gray = cv2.cvtColor(hf_patch.astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 2:3].astype(np.float32) / 255.0
    hf_gray_3 = np.repeat(hf_gray, 3, axis=-1)

    # Stronger condition injection while still keeping target cloth color as the base.
    guide_base = base_patch * 0.86 + guide_gray_3 * 0.14
    line_hint = guide_base * (1.0 - 0.42 * hf_gray_3) + 255.0 * (0.42 * hf_gray_3)

    # Let the defect prior dominate more strongly, especially where hf is bright.
    alpha = np.clip(prior_box_3 * (0.18 + 0.72 * hf_gray_3), 0.0, 0.72)
    fused_patch = guide_base * (1.0 - alpha) + line_hint * alpha

    collage = cropped_target_image.copy().astype(np.float32)
    collage[y1:y2, x1:x2, :] = fused_patch

    collage_mask = np.zeros_like(cropped_target_image, dtype=np.float32)
    collage_mask[:, :, 0] = prior_soft
    collage_mask[:, :, 1] = prior_soft
    collage_mask[:, :, 2] = prior_soft
    return collage.astype(np.uint8), collage_mask.astype(np.float32), (y1, y2, x1, x2), prior_soft


def process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_windows=3, span_ratio_thresh=0.93, aspect_ratio_thresh=12.0):
    ref_pack = build_reference_windows(
        ref_image,
        ref_mask,
        max_windows=max_windows,
        span_ratio_thresh=span_ratio_thresh,
        aspect_ratio_thresh=aspect_ratio_thresh,
    )

    tar_box_yyxx_crop = build_target_crop_box(tar_image, tar_mask, ref_pack['stats'])
    y1, y2, x1, x2 = tar_box_yyxx_crop
    cropped_target_image = tar_image[y1:y2, x1:x2, :].copy()
    tar_mask_crop = tar_mask[y1:y2, x1:x2].astype(np.uint8)

    collage, collage_mask, prior_box, prior_soft = compose_control_hint_from_target_crop(cropped_target_image, tar_mask_crop, ref_pack)
    segment_maps = build_target_segment_maps(tar_mask_crop, ref_pack, prior_soft, out_size=512)

    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value=0, random=False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=0, random=False).astype(np.float32)

    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    collage = cv2.resize(collage, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    collage_mask = cv2.resize(collage_mask, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    collage_mask = np.clip(collage_mask, 0.0, 1.0)

    if len(ref_pack['windows']) == 0:
        detail_refs = np.zeros((0, 224, 224, 3), dtype=np.float32)
        legacy_refs = ref_pack['global_ref_224'][None, ...].astype(np.float32) / 255.0
        legacy_weights = np.ones((1,), dtype=np.float32)
        legacy_meta = np.array([[0.0, 1.0, 0.5, 1.0]], dtype=np.float32)
    else:
        detail_refs = np.stack([w['ref_224'].astype(np.float32) / 255.0 for w in ref_pack['windows']], axis=0)
        legacy_refs = detail_refs.copy()
        legacy_weights = ref_pack['weights'].copy()
        legacy_meta = ref_pack['window_meta'].copy()
    global_ref = ref_pack['global_ref_224'].astype(np.float32) / 255.0
    jpg = cropped_target_image / 127.5 - 1.0
    hint = collage / 127.5 - 1.0
    hint = np.concatenate([hint, collage_mask[:, :, :1]], axis=-1)

    target_prior_full = np.zeros(tar_image.shape[:2], dtype=np.float32)
    target_prior_full[y1:y2, x1:x2] = cv2.resize(prior_soft.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

    return {
        'global_ref': global_ref.copy(),
        'detail_refs': detail_refs.copy(),
        'ref_windows': legacy_refs.copy(),
        'ref_weights': legacy_weights.copy(),
        'ref_window_meta': legacy_meta.copy(),
        'detail_reliability': ref_pack['window_reliability'].copy(),
        'detail_assignments': segment_maps['assignments'].copy(),
        'region_weight': segment_maps['region'].copy(),
        'detail_weight': segment_maps['detail'].copy(),
        'alpha_detail': np.array([ref_pack['alpha_detail']], dtype=np.float32),
        'jpg': jpg.copy(),
        'hint': hint.copy(),
        'extra_sizes': np.array([H1, W1, H2, W2]),
        'tar_box_yyxx_crop': np.array(tar_box_yyxx_crop),
        'debug': {
            'mode': ref_pack['mode'],
            'aspect_ratio': float(ref_pack['stats']['aspect_ratio']),
            'span_ratio': float(ref_pack['stats']['span_ratio']),
            'alpha_detail': float(ref_pack['alpha_detail']),
            'global_ref_224': ref_pack['global_ref_224'].copy(),
            'rep_ref_224': ref_pack['rep_ref_224'].copy(),
            'rep_def_224': ref_pack['rep_def_224'].copy(),
            'rep_guidance_224': ref_pack['rep_guidance_224'].copy(),
            'rep_hf_224': ref_pack['rep_hf_224'].copy(),
            'debug_strip': ref_pack['debug_strip'].copy(),
            'collage_rgb': collage.copy().astype(np.uint8),
            'target_prior_full': target_prior_full.copy(),
            'target_region_512': segment_maps['region'].copy(),
            'target_detail_512': segment_maps['detail'].copy(),
            'detail_assignments': segment_maps['assignments'].copy(),
            'detail_reliability': ref_pack['window_reliability'].copy(),
            'windows': [{'start': int(w['start']), 'end': int(w['end']), 'score': int(w['score'])} for w in ref_pack['windows']],
        }
    }


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop, tar_mask=None):
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2), interpolation=cv2.INTER_LINEAR)
    m = 2

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :] if pad2 > 0 else pred[:, pad1:, :]
    elif H1 < H2:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :] if pad2 > 0 else pred[pad1:, :, :]

    pred = pred[m:-m, m:-m, :]
    h = max(1, (y2 - y1) - 2 * m)
    w = max(1, (x2 - x1) - 2 * m)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

    gen_image = tar_image.copy().astype(np.float32)
    target_patch = gen_image[y1 + m:y2 - m, x1 + m:x2 - m, :].copy().astype(np.uint8)
    pred_patch = np.clip(pred, 0, 255).astype(np.uint8)

    if tar_mask is not None:
        mask_patch = tar_mask[y1 + m:y2 - m, x1 + m:x2 - m].astype(np.float32)
        alpha = build_blend_alpha(mask_patch, ksize=21, core=0.92, edge=0.03)
        pred_patch = color_match_lab_l_only_rgb(pred_patch, target_patch, mask=mask_patch)
    else:
        alpha = np.ones((h, w, 1), dtype=np.float32) * 0.78
        pred_patch = color_match_lab_l_only_rgb(pred_patch, target_patch)

    blended = target_patch.astype(np.float32) * (1.0 - alpha) + pred_patch.astype(np.float32) * alpha
    gen_image[y1 + m:y2 - m, x1 + m:x2 - m, :] = blended
    return np.clip(gen_image, 0, 255).astype(np.uint8)


def encode_reference_condition(ref_image_224):
    runtime_model, _ = load_inference_model()
    if ref_image_224.ndim == 3:
        ref_tensor = torch.from_numpy(ref_image_224.copy()).float().cuda().unsqueeze(0)
    else:
        ref_tensor = torch.from_numpy(ref_image_224.copy()).float().cuda()
    if ref_tensor.shape[-1] == 3:
        ref_tensor = einops.rearrange(ref_tensor, 'b h w c -> b c h w').contiguous()
    with torch.no_grad():
        return runtime_model.get_learned_conditioning(ref_tensor)


def get_fused_conditioning_from_windows(window_refs, window_weights, window_meta=None):
    runtime_model, _ = load_inference_model()
    if window_refs.shape[0] == 0:
        zeros = torch.zeros((1, 3, 224, 224), device=runtime_model.device)
        with torch.no_grad():
            return runtime_model.get_learned_conditioning(zeros)
    ref_tensor = torch.from_numpy(window_refs.copy()).float().cuda().unsqueeze(0)
    weight_tensor = torch.from_numpy(window_weights.astype(np.float32)).cuda().unsqueeze(0)
    cond_input = {
        'ref_windows': ref_tensor,
        'ref_weights': weight_tensor,
    }
    if window_meta is not None:
        cond_input['ref_window_meta'] = torch.from_numpy(window_meta.astype(np.float32)).cuda().unsqueeze(0)
    with torch.no_grad():
        return runtime_model.get_learned_conditioning(cond_input)


def _to_latent_weight(map_2d, device, channels=4):
    tensor = torch.from_numpy(map_2d.astype(np.float32)).to(device).view(1, 1, map_2d.shape[0], map_2d.shape[1])
    tensor = torch.nn.functional.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    if channels > 1:
        tensor = tensor.repeat(1, channels, 1, 1)
    return tensor


def build_ram_dvr_guidance(item, runtime_model, control, max_active_windows=4):
    global_cond = encode_reference_condition(item['global_ref'])
    detail_refs = item['detail_refs']
    alpha_detail = float(item.get('alpha_detail', np.array([0.0], dtype=np.float32))[0])

    if detail_refs.shape[0] == 0 or alpha_detail <= 1e-5:
        return global_cond, None

    num_detail = min(int(detail_refs.shape[0]), int(max_active_windows))
    detail_refs = detail_refs[:num_detail]
    detail_assignments = item['detail_assignments'][:num_detail]
    reliability = item['detail_reliability'][:num_detail].astype(np.float32)

    detail_tensor = torch.from_numpy(detail_refs.copy()).float().cuda()
    detail_tensor = einops.rearrange(detail_tensor, 'n h w c -> n c h w').contiguous()
    with torch.no_grad():
        detail_conds = runtime_model.get_learned_conditioning(detail_tensor)

    device = control.device
    assignment_tensors = []
    for i in range(num_detail):
        assignment_tensors.append(_to_latent_weight(detail_assignments[i], device, channels=4))
    assignments = torch.cat(assignment_tensors, dim=0)
    region = _to_latent_weight(item['region_weight'], device, channels=4)
    detail = _to_latent_weight(item['detail_weight'], device, channels=4)

    rel = torch.from_numpy(reliability).to(device=device, dtype=control.dtype).view(num_detail, 1, 1, 1)
    ram_dvr = {
        'detail_crossattn': detail_conds,
        'assignments': assignments.to(dtype=control.dtype),
        'region_weight': region.to(dtype=control.dtype),
        'detail_weight': detail.to(dtype=control.dtype),
        'reliability': rel,
        'alpha_detail': float(np.clip(alpha_detail, 0.0, 1.0)),
        'max_residual_norm': 1.35,
    }
    return global_cond, ram_dvr


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale=5.0, max_windows=3,
                           span_ratio_thresh=0.93, aspect_ratio_thresh=12.0, return_debug=False):
    runtime_model, runtime_sampler = load_inference_model()
    item = process_pairs(
        ref_image,
        ref_mask,
        tar_image,
        tar_mask,
        max_windows=max_windows,
        span_ratio_thresh=span_ratio_thresh,
        aspect_ratio_thresh=aspect_ratio_thresh,
    )

    if save_memory:
        runtime_model.low_vram_shift(is_diffusing=False)

    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda()
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    global_cond, ram_dvr = build_ram_dvr_guidance(item, runtime_model, control)

    guess_mode = False
    H, W = 512, 512
    cond = {
        'c_concat': [control],
        'c_crossattn': [global_cond],
    }
    if ram_dvr is not None:
        cond['ram_dvr'] = ram_dvr

    zeros = torch.zeros((1, 3, 224, 224), device=control.device)
    un_cond = {
        'c_concat': None if guess_mode else [control],
        'c_crossattn': [runtime_model.get_learned_conditioning(zeros)],
    }
    shape = (4, H // 8, W // 8)

    if save_memory:
        runtime_model.low_vram_shift(is_diffusing=True)

    strength = 1.0
    ddim_steps = RUNTIME_DDIM_STEPS
    scale = guidance_scale
    eta = 0.0

    target_latent = encode_target_latent(runtime_model, item['jpg'])
    edit_weight_latent, anchor_weight_raw = prepare_tla_masks(
        tar_mask,
        item['tar_box_yyxx_crop'],
        latent_size=H // 8,
    )
    anchor_weight_latent = protect_tla_anchor(
        anchor_weight_raw,
        protection_maps=[item.get('region_weight'), item.get('detail_weight')],
        protect_strength=0.92,
    )
    runtime_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
    x_T, anchor_noise = build_tla_initial_noise(
        (num_samples, *shape),
        runtime_sampler,
        target_latent,
        anchor_weight_latent,
        device=control.device,
    )
    correction_fn = TargetLatentAnchor(
        runtime_sampler,
        target_latent,
        anchor_noise,
        anchor_weight_latent,
        strength=0.42,
    )

    runtime_model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    samples, _ = runtime_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
        x_T=x_T,
        latent_correction_fn=correction_fn,
    )

    if save_memory:
        runtime_model.low_vram_shift(is_diffusing=False)

    x_samples = runtime_model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

    pred = np.clip(x_samples[0], 0, 255)[1:, :, :]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop']
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop, tar_mask=tar_mask)

    if return_debug:
        debug = item['debug']
        debug['tar_box_yyxx_crop'] = item['tar_box_yyxx_crop'].copy()
        debug['target_crop_512'] = np.clip((item['jpg'].copy() + 1.0) * 127.5, 0, 255).astype(np.uint8)
        debug['tla_edit_weight_64'] = edit_weight_latent.copy()
        debug['tla_anchor_weight_raw_64'] = anchor_weight_raw.copy()
        debug['tla_anchor_weight_protected_64'] = anchor_weight_latent.copy()
        return gen_image, debug
    return gen_image
