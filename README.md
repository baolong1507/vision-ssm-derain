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

## Results
| Dataset | PSNR ↑ | SSIM ↑ |
|---|---:|---:|
| Rain200L |  |  |
| Rain200H |  |  |
| Rain1400 |  |  |
| DID-Data |  |  |

Qualitative examples are in `results/qualitative/`.

## Setup
```bash
pip install -r requirements.txt

## Data preparation

> **Note**: Most deraining datasets are distributed under specific licenses.
> This repo **does not** redistribute datasets. Please download them from official sources and place them in the folder structure below.

### Recommended directory layout

Create a data root (e.g., `data/`) and organize datasets like:

```text
data/
  Rain200L/
    train/
      rainy/
      gt/
    test/
      rainy/
      gt/
  Rain200H/
    train/
      rainy/
      gt/
    test/
      rainy/
      gt/
  Rain1400/
    train/
      rainy/
      gt/
    test/
      rainy/
      gt/
  DID-Data/
    train/
      rainy/
      gt/
    test/
      rainy/
      gt/