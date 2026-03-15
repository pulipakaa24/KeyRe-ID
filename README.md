# KeyRe-ID (Fork)

> **This is a fork of [JinSeong0115/KeyRe-ID](https://github.com/JinSeong0115/KeyRe-ID).**
> The original repository introduces a keypoint-guided video person re-identification framework using a Vision Transformer backbone with temporal attention weighting.
> This fork makes targeted updates to the heatmap generation and loading pipelines to improve correctness, dataset compatibility, and training stability.

---

## Changes from Upstream

### Heatmap Generation (`keypoint/keypoint_to_mask.py`)

- Refactored into a reusable `KeypointsToMasks` class with a cached Gaussian kernel (computed once per output size, not per frame).
- Gaussian heatmap placement is now boundary-aware: the kernel is correctly clipped when a keypoint lies near the edge of the image, preventing index errors.
- Low-confidence keypoints are skipped per joint rather than zeroing the entire group.
- Added an argparse CLI so heatmap generation can be run directly as a script for both `bbox_train` and `bbox_test` phases.

### Heatmap Loading (`heatmap_loader.py`)

- Added `_resolve_heatmap_path()` to correctly map image frame paths to `.npy` heatmap files for both **MARS** (`bbox_train/personID/frame`) and **iLIDS-VID** (`cam1|cam2/personID/frame`) directory structures.
- `Heatmap_Dataset` (test/val) reconstructs the exact clip frame indices used by `VideoDataset` so heatmaps are guaranteed to align with the sampled frames, including repeated frames in the final clip.
- `Heatmap_Dataset_inderase` (train) applies the same geometric augmentations to heatmaps as to images: horizontal flip, random crop, and random erasing — keeping heatmap and RGB inputs spatially consistent.
- Per-channel min-max scaling is applied before normalization, preventing channels with near-zero responses from contributing noise.
- An in-memory heatmap cache in `Heatmap_Dataset_inderase` avoids redundant disk reads for frequently sampled identities.

---

## Setup

```bash
conda create -n KeyReID python=3.8
conda activate KeyReID
pip install torch torchvision timm openpifpaf torch-ema tqdm scipy
```

## Data Preparation

### 1. Extract keypoints

```bash
python keypoint/extract_keypoint.py --dataset_path /path/to/MARS
```

### 2. Generate heatmaps

```bash
# MARS
python keypoint/keypoint_to_mask.py \
    --dataset_path /path/to/MARS \
    --output_dir /path/to/MARS/heatmap

# iLIDS-VID
python keypoint/keypoint_to_mask.py \
    --dataset_path /path/to/iLIDSVID \
    --output_dir /path/to/iLIDSVID/heatmap
```

Expected heatmap directory layout:

```
# MARS
heatmap/
  bbox_train/
    0001/
      0001C1T0001F001.npy
      ...
  bbox_test/
    ...

# iLIDS-VID
heatmap/
  cam1/
    person_001/
      frame_001.npy
      ...
  cam2/
    ...
```

## Training

```bash
python train.py --dataset iLIDSVID --data_root /path/to/data
```

## Evaluation

```bash
python test.py --dataset iLIDSVID --data_root /path/to/data --weights /path/to/best_CMC.pth
```

---

## Citation

If you use this work, please cite the original paper:

```
@inproceedings{KeyReID,
  title     = {KeyRe-ID: Keypoint-Guided Video Person Re-Identification},
  author    = {Jin Seong and others},
  booktitle = {..},
  year      = {..}
}
```
