# video2sim-cup

**Video2Sim — Ceramic Mug | CSCI 5961 AI Capstone, Fall 2026**  
Saint Louis University · Sidhvi Nuvvula

Converts a handheld smartphone video of a ceramic coffee mug into a simulation-ready USD asset using a six-stage pipeline: FFmpeg → COLMAP → DA3 → SAM3 → HoloScene → USD. The final asset is compatible with NVIDIA Isaac Sim and any USD-compliant simulator.

> **Partner repository:** [Chandana9700/video2sim-tree](https://github.com/Chandana9700/video2sim-tree) — tree dataset (same pipeline, different object).

---

## Repository Structure

```
video2sim-cup/
├── src/
│   ├── da3/              # Depth Anything v3 module code
│   ├── sam3/             # SAM3 / SAM2 module code
│   └── holoscene/        # HoloScene reconstruction module
├── scripts/
│   ├── batch_reconstruct.bat   # Windows: FFmpeg + COLMAP automation
│   ├── run_da3.sh              # SLURM job: DA3 depth estimation
│   ├── run_sam3.sh             # SLURM job: SAM3 instance segmentation
│   ├── run_holoscene.sh        # SLURM job: HoloScene reconstruction
│   └── run.sh                  # HPC: chain all three SLURM jobs
├── docs/
│   ├── pipeline_architecture.png
│   └── holoscene_patch.md      # Required HoloScene code patch
├── experiments/
│   ├── configs/                # SLURM job configs and hyperparameters
│   ├── logs/                   # SLURM output logs
│   └── figures/                # Generated evaluation figures
├── reports/                    # Weekly progress reports
├── poster/                     # poster.pdf + PowerPoint source
├── report/                     # final_report.pdf + LaTeX source
├── data/                       # NOT committed — see Data section below
│   └── input/custom/cup/
│       ├── images/             # JPEG frames (FFmpeg output)
│       ├── sparse/0/           # COLMAP TXT model
│       ├── transforms.json     # DA3 output
│       ├── instance_mask/      # SAM3 output
│       └── prompts.txt         # SAM3 text prompts
├── environment_da3.yml         # Conda env for DA3 + SAM3
├── environment_holoscene.yml   # Conda env for HoloScene
├── requirements.txt            # Python dependencies (pip-compatible)
└── README.md
```

---

## Object

A tall white ceramic coffee mug with a distressed horizontal black line pattern, yellow circular graphic, bold "CAFÉ" text, a small coffee bean icon, and a slender curved white handle. Captured with a handheld smartphone performing a 360° rotation around the object under indoor ambient lighting.

**SAM3 prompts used** (`prompts.txt`):
```
Tall white ceramic coffee mug.
Distressed horizontal black line pattern.
Yellow circular graphic on the side.
Bold black "CAFÉ" text.
Small coffee bean icon.
Slender curved white handle.
Cylindrical shape with flat base.
```

---

## Prerequisites

### Local Machine (Windows 10/11)

| Tool | Version | Install path |
|------|---------|-------------|
| FFmpeg | 7.x | `03 FFMPEG\` or `03 FFMPEG\bin\` |
| COLMAP | 4.0.2 | `01 COLMAP\` or `01 COLMAP\bin\` |
| MeshLab | 2023+ | Standard system install |

Folder layout expected by `batch_reconstruct.bat`:
```
01 COLMAP\
02 VIDEOS\
03 FFMPEG\
04 SCENES\
batch_reconstruct.bat   ← place here, one level above the four folders
```

### HPC Cluster (SLU Libra or equivalent SLURM cluster)

- SLURM scheduler
- NVIDIA H100 NVL or A100 for DA3 (80 GB VRAM recommended)
- NVIDIA L40S or A100 for SAM3 and HoloScene (40+ GB VRAM)
- Conda (Miniconda or Anaconda)
- Hugging Face account with access to `GonzaloMG/marigold-e2e-ft-normals`

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sidhvi16/video2sim-cup.git
cd video2sim-cup
```

### 2. Create conda environments (HPC)

```bash
# DA3 + SAM3 environment
conda env create -f environment_da3.yml
conda activate da3_env

# HoloScene environment (must be compiled on a GPU compute node)
# SSH into a GPU node first, then:
conda env create -f environment_holoscene.yml
conda activate holoscene_env
```

### 3. Apply the HoloScene patch

Before running any HoloScene jobs, apply the instance mesh fallback patch:

```bash
cd src/holoscene/training/
cp holoscene_train.py holoscene_train.py.bak

# Open holoscene_train.py and find instance_meshes_post_pruning (~line 529).
# Replace:
#   assert os.path.exists(obj_i_mesh_path), f"mesh {obj_i} does not exist"
# With the fallback block in docs/holoscene_patch.md
```

See `docs/holoscene_patch.md` for the full patch and explanation.

### 4. Export your Hugging Face token

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Re-export this in every new shell session before submitting SLURM jobs.

---

## Running the Pipeline

### Stage 1 & 2 — Local (Windows): Frame Extraction + COLMAP

1. Place your video file in `02 VIDEOS\`.
2. Double-click `scripts\batch_reconstruct.bat` (or run from Command Prompt).
3. The script extracts frames at `EXTRACT_FPS=1.3` (≈335 frames for the cup video) and runs COLMAP feature extraction, sequential matching, sparse mapping, and TXT export automatically.
4. After completion, open the COLMAP GUI and verify camera frustums cover 360° with no gaps.

### Stage 2.5 — Local: Point Cloud Cleaning (MeshLab)

```
File > Import Mesh > sparse\0\points3D.ply
Filters > Cleaning and Repairing > Remove Isolated Pieces (wrt Diameter, 10% threshold)
Filters > Cleaning and Repairing > Remove Duplicate Vertices
File > Export Mesh As > points3D_clean.ply
```

### Stage 3 — Transfer to HPC

```bash
# From Windows terminal (adjust paths as needed)
scp -r .\images <username>@libra.slu.edu:~/cup1/data/input/custom/cup/
scp -r .\sparse\0\*.txt <username>@libra.slu.edu:~/cup1/data/input/custom/cup/sparse/0/
scp points3D_clean.ply <username>@libra.slu.edu:~/cup1/
```

### Stage 4, 5, 6 — HPC: DA3 → SAM3 → HoloScene

Option A — run individually:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
cd ~/cup1

sbatch scripts/run_da3.sh
# Wait for completion, then:
sbatch scripts/run_sam3.sh
# Wait for completion, then:
sbatch scripts/run_holoscene.sh
```

Option B — chain automatically with SLURM dependencies:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
cd ~/cup1
bash scripts/run.sh
```

Monitor jobs:

```bash
squeue -u $USER
tail -f experiments/logs/da3_cup_<jobid>.log
tail -f experiments/logs/sam3_cup_<jobid>.log
tail -f experiments/logs/holoscene_cup_<jobid>.log
```

### Stage 7 — USD Validation

Download the USD file:

```bash
scp <username>@libra.slu.edu:~/cup1/repo/modules/holoscene/exps/holoscene_cup/*/plots/*.usd* .
```

Open in NVIDIA Isaac Sim: `File > Open > select .usd or .usdc`. Verify:
- `metersPerUnit = 0.01`
- `upAxis = Y`
- No import errors in the console
- Mesh geometry covers the full mug body and handle

---

## Expected Outputs

| File | Location | Description |
|------|----------|-------------|
| `frame_NNNNNN.jpg` | `data/input/custom/cup/images/` | ≈335 JPEG frames |
| `transforms.json` | `data/input/custom/cup/` | DA3 camera poses + depth |
| `instance_mask/*.png` | `data/input/custom/cup/instance_mask/` | Per-frame binary masks |
| `surface_100_whole.ply` | HoloScene exps dir | Whole-scene Poisson mesh |
| `cup.usd` / `cup.usdc` | HoloScene exps dir | Final USD asset |

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `EXTRACT_FPS` | 1.3 | Yields ≈335 frames for a 4m18s video |
| COLMAP `--SequentialMatching.overlap` | 50 | Frames matched per direction |
| COLMAP `--Mapper.min_num_matches` | 10 | Lowered from default 15 |
| `SAM3_MIN_SCORE` | 0.10 | Minimum mask confidence |
| `SAM3_MIN_FRAME_DURATION` | 40 | Minimum frames object must appear |
| HoloScene iterations | 256 | Gaussian splatting training steps |

---

## Hardware Requirements Summary

| Stage | Hardware | Approx. Time |
|-------|----------|-------------|
| FFmpeg + COLMAP | NVIDIA Quadro M3000M (local) | ~47 min |
| DA3 | NVIDIA H100 NVL (HPC) | ~1.5 hr |
| SAM3 | NVIDIA L40S (HPC) | ~1 hr |
| HoloScene | NVIDIA L40S (HPC) | ~3.5 hr |

---

## Troubleshooting

**DA3 fails with Hugging Face 401 error:** Re-export `HF_TOKEN` in the current shell before submitting.

**HoloScene crashes in post-processing with `AssertionError: mesh N does not exist`:** The HoloScene patch has not been applied. See `docs/holoscene_patch.md`.

**COLMAP registration rate < 80%:** Reduce `EXTRACT_FPS` to get more frames, or increase `--SequentialMatching.overlap`.

**SAM3 masks miss the object:** Edit `data/input/custom/cup/prompts.txt` with more specific visual descriptors and resubmit `run_sam3.sh`.

---

## Citation

If you use this pipeline in your work, please cite:

```bibtex
@misc{video2sim_cup_2026,
  title  = {Video2Sim: Cup Dataset Pipeline},
  author = {Nuvvula, Sidhvi},
  year   = {2026},
  note   = {\url{https://github.com/Sidhvi16/video2sim-cup}}
}
```

---

## License

This project is submitted as coursework for CSCI 5961 – AI Capstone – Fall 2026, Saint Louis University. Third-party tools (COLMAP, DA3, SAM3, HoloScene) retain their respective licenses.
