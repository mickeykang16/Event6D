# Event6D: Event-based Novel Object 6D Pose Tracking

[Jae-Young Kang](https://mickeykang16.github.io/)\*, [Hoonehee Cho](https://chohoonhee.github.io/hoonheecho/)\*, [Taeyeop Lee](https://sites.google.com/view/taeyeop-lee/)\*, [Minjun Kang](https://sites.google.com/view/minjun-kang), [Bowen Wen](https://wenbowen123.github.io/), [Youngho Kim](https://scholar.google.com/citations?user=ZDpIMQ0AAAAJ&hl=en), [Kuk-Jin Yoon](https://scholar.google.com/citations?user=1NvBj_gAAAAJ&hl=en)

KAIST &nbsp;|&nbsp; NVIDIA

\* Equal contribution

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2603.28045)

![Demo](demo.gif)

## Changelog🔥

- [2026/05] Public release: code, pretrained weights, and two evaluation datasets.
- [2026/03/26] Repository created.

## Setup

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/mickeykang16/Event6D.git
cd Event6D

# 2. Conda env (PyTorch 2.1.1 + CUDA 11.8)
conda create -n event6d python=3.9 -y
conda activate event6d
pip install -r requirements.txt
pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt211/download.html
pip install -e bop_toolkit

# 3. Build C++/CUDA extensions
bash build_all_conda.sh
```

## Data and weights

All released artifacts are hosted on Hugging Face. Download with `huggingface-cli`:

```bash
# Pretrained checkpoints (≈108 MB)
huggingface-cli download mickeykang/Event6D-weights --local-dir ./weights

# Event6D — real-world capture (≈19 GB)
huggingface-cli download mickeykang/Event6D --repo-type dataset \
    --local-dir ./data/Event6D

# EventHO3D — HO3D-v2 event-augmented evaluation (≈3 GB)
huggingface-cli download mickeykang/EventHO3D --repo-type dataset \
    --local-dir ./data/EventHO3D
```

For HO3D evaluation, also obtain the original [HO3D-v2 evaluation split](https://www.tugraz.at/index.php?id=40231)
and merge it under `./data/EventHO3D/evaluation/`. Place the [YCB-Video models](https://rse-lab.cs.washington.edu/projects/posecnn/)
under `./data/EventHO3D/ycb_models/` (or set `YCB_MODELS_PATH=/path/to/ycb_models`).

After all downloads, the layout should be:

```
Event6D/
├── weights/
│   ├── depth_extrapolation.pth
│   ├── e2vid.pth.tar
│   └── foundationpose_refiner/
│       ├── config.yml
│       └── model.pth
├── data/
│   ├── Event6D/        # real-world capture
│   └── EventHO3D/      # HO3D-v2 event augmentation + your HO3D-v2 download
└── ...
```

## Evaluation

### Event6D (real-world)

```bash
# 120 fps — pose evaluated at each event sub-timestep
CUDA_VISIBLE_DEVICES=0 python3 run_event6d_tracking.py \
    --name event6d_eval_120fps \
    --video_dirs ./data/Event6D \
    --eval_fps 120 --e2vid --online \
    --ev_width 640 --ev_height 360 \
    --depth_extrapolation_ckpt ./weights/depth_extrapolation.pth

# 30 fps — pose evaluated only at RGB instants
CUDA_VISIBLE_DEVICES=0 python3 run_event6d_tracking.py \
    --name event6d_eval_30fps \
    --video_dirs ./data/Event6D \
    --eval_fps 30 --e2vid --online \
    --ev_width 640 --ev_height 360 \
    --depth_extrapolation_ckpt ./weights/depth_extrapolation.pth
```

### EventHO3D (HO3D-v2 + simulated events)

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_eventho3d_tracking.py \
    --name eventho3d_eval \
    --video_dirs ./data/EventHO3D/evaluation \
    --stride 10 --online \
    --depth_extrapolation_ckpt ./weights/depth_extrapolation.pth \
    --e2vid_ckpt ./weights/e2vid.pth.tar
```

### Output

Per-run metrics are written to `outputs/<name>/0_eval_metric/`:
- `<sequence>_*.xlsx` — per-sequence ADDS / ADD / AR / MSSD / MSPD / VSD / IoU
- `0_mean_all.xlsx` — aggregated table; row `ALL` is the dataset-wide mean

> The dataloader caches voxel grids / E2VID inputs on the first run (see
> [Disk-space note](#disk-space-note-event-caches) below). Subsequent runs reuse them.

## Training data (Blender)

The released `depth_extrapolation.pth` checkpoint was trained on Blender-rendered
sequences with Google Scanned Objects:

```bash
# Primary (easy subset, ≈255 GB) — what the released checkpoint actually consumed
huggingface-cli download mickeykang/Event6DBlender --repo-type dataset \
    --local-dir ./data/Event6DBlender

# Optional extension (medium subset, ≈483 GB) — extra sequences, not used yet
huggingface-cli download mickeykang/Event6DBlenderMedium --repo-type dataset \
    --local-dir ./data/Event6DBlender
```

Both downloads merge into a single tree:
- `train.txt` / `test.txt` — split lists
- `gso/` — 1035 GSO meshes (CC-BY 4.0)
- `EvBlenderProc/{easy,medium}_9/` — RGB + depth + meta
- `EvBlenderProcEv/{easy,medium}_9/` — raw event NPZ

For reproduction only the `easy` subset (≈255 GB) is required — the training dataloader
hardcodes `categories=['easy']`.

## Disk-space note (event caches)

The first time you run **either** evaluation or training, the dataloader materializes
voxel-grid + E2VID-input caches alongside the raw events. Expect roughly:

| Where | Approx. extra disk |
|---|---|
| Event6D real-world (`./data/Event6D/<seq>/<run>/parsed_voxel_*`, `parsed_e2vid*`) | ≈1 GB per sequence, ≈15 GB total (14 sequences) |
| EventHO3D (`./data/EventHO3D/evaluation_voxel_*`) | ≈0.5 GB per sequence, ≈7 GB total (13 sequences) |
| Event6DBlender training (`./data/Event6DBlender/EvBlenderProcEv_cache/`) | ≈90 GB across the full split |

## Training code

**Coming soon** — the end-to-end training pipeline (refiner + depth-extrapolation +
E2VID) is being prepared for public release. For now this repository contains evaluation
code only.

## Acknowledgements

This work builds on:
- [FoundationPose](https://github.com/NVlabs/FoundationPose) — pose refiner backbone
- [E2VID](https://github.com/uzh-rpg/rpg_e2vid) — event-to-video reconstruction
- [CBMNet](https://github.com/intelpro/CBMNet) — cross-modal bilateral mutual network used in our depth-extrapolation stage

## Cite this work📝

```
@inproceedings{kang2026event6d,
  title     = {Event6D: Event-based Novel Object 6D Pose Tracking},
  author    = {Kang, Jae-Young and
               Cho, Hoonehee and
               Lee, Taeyeop and
               Kang, Minjun and
               Wen, Bowen and
               Kim, Youngho and
               Yoon, Kuk-Jin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
