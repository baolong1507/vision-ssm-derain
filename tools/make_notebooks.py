from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

def nb(title, intro_md, code_cells):
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n{intro_md}"))
    for c in code_cells:
        if c["type"] == "md":
            cells.append(nbf.v4.new_markdown_cell(c["content"]))
        else:
            cells.append(nbf.v4.new_code_cell(c["content"]))
    nb["cells"] = cells
    return nb

COMMON_SETUP = r"""
# === 0) Setup ===
import os, sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parents[0].parents[0]  # notebooks/ -> repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("PROJECT_ROOT:", PROJECT_ROOT)
"""

INSTALL = r"""
# (Colab) Uncomment if needed:
# !pip -q install -r ../requirements.txt
"""

CHECK_DATA = r"""
from omegaconf import OmegaConf
data_cfg = OmegaConf.load("../configs/data_config.yaml")
print(data_cfg)

# sanity check paths
from pathlib import Path
root = Path(data_cfg.data_root)
for d in data_cfg.datasets:
    p = root / d
    print(d, "->", p, "exists:", p.exists())
"""

RUN_PHASE1 = r"""
# === 1) Train Phase 1 ===
!python ../scripts/train_phase1.py --cfg ../configs/phase1.yaml --data ../configs/data_config.yaml
"""

RUN_PHASE2 = r"""
# === 1) Train Phase 2 ===
!python ../scripts/train_phase2.py --cfg ../configs/phase2.yaml --data ../configs/data_config.yaml
"""

RUN_PHASE3 = r"""
# === 1) Train Phase 3 ===
!python ../scripts/train_phase3.py --cfg ../configs/phase3.yaml --data ../configs/data_config.yaml
"""

def write_nb(filename, notebook):
    out = NB_DIR / filename
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    print("Wrote:", out)

# ----------------- Phase 1 -----------------
phase1 = nb(
    "01_phase1_baseline_unet",
    intro_md="""
**Goal:** Establish a strong, stable baseline (UNet) for fair comparison.

**Pipeline:** Dataset → UNet → L1/Charbonnier → PSNR/SSIM evaluation (PSNR is main monitor).
""",
    code_cells=[
        {"type":"code","content":INSTALL},
        {"type":"code","content":COMMON_SETUP},
        {"type":"md","content":"## Data configuration & sanity check"},
        {"type":"code","content":CHECK_DATA},
        {"type":"md","content":"## Train"},
        {"type":"code","content":RUN_PHASE1},
        {"type":"md","content":"## Notes / Discussion\n- Check outputs under `outputs/phase1/`.\n- Use best checkpoint (highest val/psnr) for later comparison."},
    ]
)
write_nb("01_phase1_baseline_unet.ipynb", phase1)

# ----------------- Phase 2 -----------------
phase2 = nb(
    "02_phase2_fess_unet",
    intro_md="""
**Goal:** Improve detail restoration & generalization using frequency-enhanced skip learning (FESS-UNet).

**Key changes vs Phase 1:**
- Frequency enhancement on skip connections
- Add SSIM + FFT amplitude loss
""",
    code_cells=[
        {"type":"code","content":INSTALL},
        {"type":"code","content":COMMON_SETUP},
        {"type":"md","content":"## Data configuration & sanity check"},
        {"type":"code","content":CHECK_DATA},
        {"type":"md","content":"## Train"},
        {"type":"code","content":RUN_PHASE2},
        {"type":"md","content":"## Notes / Discussion\n- Compare PSNR gain vs Phase 1.\n- Inspect qualitative samples (blur, edges, heavy rain)."},
    ]
)
write_nb("02_phase2_fess_unet.ipynb", phase2)

# ----------------- Phase 3 -----------------
phase3 = nb(
    "03_phase3_fessm",
    intro_md="""
**Goal:** Add SSM-inspired global modeling (FESSM) for better long-range dependency handling.

**Key changes vs Phase 2:**
- Insert SSM2D blocks into bottleneck + decoder stages
- Keep frequency enhancement + multi-loss (L1 + SSIM + FFT)
""",
    code_cells=[
        {"type":"code","content":INSTALL},
        {"type":"code","content":COMMON_SETUP},
        {"type":"md","content":"## Data configuration & sanity check"},
        {"type":"code","content":CHECK_DATA},
        {"type":"md","content":"## Train"},
        {"type":"code","content":RUN_PHASE3},
        {"type":"md","content":"## Notes / Discussion\n- Try `ssm_mode: convscan` first.\n- If you want more “state update” behavior, set `ssm_mode: recurrent` (slower)."},
    ]
)
write_nb("03_phase3_fessm.ipynb", phase3)
