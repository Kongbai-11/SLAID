# SLAID

SLAID is an inference pipeline for local defect synthesis based on the AnyDoor/ControlNet diffusion interface. The project keeps the original model backbone interface and adds three training-free inference algorithms:

- **SSRM**: shape-aware reference selection and region mapping for local or elongated defects.
- **AHFG**: adaptive high-frequency guidance for defect detail transfer.
- **TLA**: target-domain latent anchoring to preserve the target background outside the editable region.

## Installation

Create an environment and install dependencies:

```bash
conda env create -f environment.yaml
conda activate SLAID
pip install -r requirements.txt
```

The exact CUDA, PyTorch and xFormers versions may need to match your GPU driver.

## Model Weights

Model weights are not tracked by Git. Download the AnyDoor checkpoint from [ModelScope](https://modelscope.cn/models/iic/AnyDoor/files).

## Dataset

The dataset is not included in this repository. Download it from [Downland Link(google drive)](https://drive.google.com/file/d/1dHDe4bnT5hAFbopLf2-zFCR7aNSXVnLL/view?usp=sharing).

## Inference

`python scripts/infer.py`

## Notes for Open Source Release

- Do not commit checkpoints, generated outputs, local datasets or IDE files.
- Keep third-party licenses for AnyDoor/ControlNet/Stable Diffusion/DINOv2-related code.
