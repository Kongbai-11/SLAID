import argparse
import sys
from pathlib import Path

import cv2
from pytorch_lightning import seed_everything


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slaid import pipeline
from slaid.io import ensure_dir, read_mask, read_rgb


def build_argparser():
    parser = argparse.ArgumentParser(description='Run SLAID inference on one reference-target pair.')
    parser.add_argument('--ref-image', required=True, help='Reference defect image.')
    parser.add_argument('--ref-mask', required=True, help='Binary mask for the reference defect.')
    parser.add_argument('--target-image', required=True, help='Target/background image.')
    parser.add_argument('--target-mask', required=True, help='Binary mask for the target edit region.')
    parser.add_argument('--output', default='outputs/slaid_result.png', help='Output image path.')
    parser.add_argument('--inference-config', default=pipeline.DEFAULT_INFERENCE_CONFIG)
    parser.add_argument('--pretrained-model', default='')
    parser.add_argument('--ddim-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--max-windows', type=int, default=3)
    parser.add_argument('--span-ratio-thresh', type=float, default=0.93)
    parser.add_argument('--aspect-ratio-thresh', type=float, default=12.0)
    parser.add_argument('--seed', type=int, default=1234)
    return parser


def main():
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    pipeline.RUNTIME_INFERENCE_CONFIG = args.inference_config
    pipeline.RUNTIME_PRETRAINED_MODEL = args.pretrained_model
    pipeline.RUNTIME_DDIM_STEPS = args.ddim_steps

    ref_image = read_rgb(args.ref_image)
    ref_mask = read_mask(args.ref_mask)
    target_image = read_rgb(args.target_image)
    target_mask = read_mask(args.target_mask)

    result = pipeline.inference_single_image(
        ref_image=ref_image,
        ref_mask=ref_mask,
        tar_image=target_image,
        tar_mask=target_mask,
        guidance_scale=args.guidance_scale,
        max_windows=args.max_windows,
        span_ratio_thresh=args.span_ratio_thresh,
        aspect_ratio_thresh=args.aspect_ratio_thresh,
    )

    output_path = Path(args.output)
    ensure_dir(str(output_path.parent))
    cv2.imwrite(str(output_path), result[:, :, ::-1])
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
