# vision-ssm-derain
Robust image deraining for autonomous driving under extreme weather, with reproducible training/evaluation and real-rain generalization.

# Image Deraining with Deep Learning for Autonomous Driving in Extreme Weather

This repository provides **reproducible training/evaluation** for robust image deraining, targeting **generalization to real rain** and **downstream driving perception readiness**.

## Highlights
- ✅ Reproducible configs (seeded splits, fixed eval)
- ✅ Metrics: PSNR / SSIM / (optional) DISTS + qualitative comparisons
- ✅ Baselines + proposed model (e.g., Vision SSM + hybrid frequency)
- ✅ Inference demo on custom images / videos (optional)

## Method (overview)
(1) Input rainy image → (2) Derain network → (3) Restored output  
Describe your key modules: hybrid spatial-frequency, SSM blocks, etc.

## Dataset

| Dataset  | Link |
|---|---|
| Rain200L | [dataset](https://pan.baidu.com/s/1Hvm9ctniC7PMQdKrI_lf3QdKrI_lf3Q) |
| Rain200H | [dataset](https://pan.baidu.com/s/1Hvm9ctniC7PMQdKrI_lf3QdKrI_lf3Q) |
| Rain1400 | [dataset](https://pan.baidu.com/s/1Hvm9ctniC7PMQdKrI_lf3QdKrI_lf3Q) |
| DID-Data | [dataset](https://pan.baidu.com/s/1Hvm9ctniC7PMQdKrI_lf3QdKrI_lf3Q) |

## Results
|         | Baseline U-Net  | FESS U-Net      | FESSM           |    
| Dataset | PSNR ↑ | SSIM ↑ | PSNR ↑ | SSIM ↑ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|
| Rain200L | 31.83  | 0.924 |
| Rain200H |  |  |
| Rain1400 |  |  |
| DID-Data |  |  |

Qualitative examples are in `results/qualitative/`.

## Setup
```bash
pip install -r requirements.txt
```

## Data preparation
This repo **does not** redistribute datasets. Please download them from official sources and place them in the folder structure below.

## Training
### Quick start
```bash
python scripts/train.py --config configs/model_phaseX_ssm.yaml
```

### Normal start
```bash
python scripts/train.py \
  --config configs/model_phaseX_ssm.yaml \
  --seed 42 \
  --batch_size 12 \
  --img_size 256 \
  --max_epochs 80
```

## Evaluation
Evaluate a trained checkpoint on all supported test sets:
```bash
python scripts/eval.py \
  --config configs/model_phaseX_ssm.yaml \
  --ckpt results/checkpoints/best.ckpt
```
Evaluate a specific dataset only:
```bash
python scripts/eval.py \
  --config configs/model_phase3_ssm.yaml \
  --ckpt results/checkpoints/best.ckpt \
  --dataset Rain200H
```