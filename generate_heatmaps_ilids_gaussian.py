"""
generate_heatmaps_ilids_gaussian.py

Generate 6-channel body-part heatmaps for iLIDS-VID using:
  - MMPose (RTMPose-L) for keypoint extraction  (works in WSL2, unlike PifPaf)
  - Original KeyRe-ID Gaussian blob method for heatmap rendering

This matches the paper's heatmap format: a small Gaussian kernel stamped at
each joint location, with element-wise max across joints within each body-part
group.  The filled-region approach used in the earlier fork produces diffuse
attention weights that hurt the KPS module.

Usage:
    python generate_heatmaps_ilids_gaussian.py --dataset_path ./data/iLIDSVID
    python generate_heatmaps_ilids_gaussian.py --dataset_path ./data/iLIDSVID --device cuda

Output layout (matches _resolve_heatmap_path in the fork's heatmap_loader.py):
    <dataset_path>/heatmap/cam1/person001/filename.npy   (shape: 6, H, W)
"""

import os
import argparse
import numpy as np
from collections import OrderedDict
from scipy.signal.windows import gaussian
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════
# Body-part groupings — identical to original keypoint_to_mask.py
# ═══════════════════════════════════════════════════════════════════════
joints_dict = OrderedDict()
joints_dict['head']      = ['nose', 'Leye', 'Reye', 'LEar', 'REar']
joints_dict['torso']     = ['LS', 'RS', 'LH', 'RH']
joints_dict['left_arm']  = ['LE', 'LW']
joints_dict['right_arm'] = ['RE', 'RW']
joints_dict['left_leg']  = ['LK', 'LA']
joints_dict['right_leg'] = ['RK', 'RA']

pose_keypoints = [
    'nose', 'Leye', 'Reye', 'LEar', 'REar',
    'LS', 'RS', 'LE', 'RE', 'LW', 'RW',
    'LH', 'RH', 'LK', 'RK', 'LA', 'RA',
]
keypoints_dict = {name: idx for idx, name in enumerate(pose_keypoints)}


# ═══════════════════════════════════════════════════════════════════════
# Gaussian heatmap utilities — verbatim from original keypoint_to_mask.py
# ═══════════════════════════════════════════════════════════════════════
def gkern(kernlen=21, std=None):
    if std is None:
        std = kernlen / 4
    gkern1d = gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def rescale_keypoints(rf_keypoints, size, new_size):
    w, h = size
    new_w, new_h = new_size
    rf_keypoints = rf_keypoints.copy()
    rf_keypoints[:, 0] = rf_keypoints[:, 0] * new_w / w
    rf_keypoints[:, 1] = rf_keypoints[:, 1] * new_h / h
    return rf_keypoints


class KeypointsToMasks:
    """Exact replica of the original paper's heatmap generator."""

    def __init__(self, g_scale=11, vis_thresh=0.1, vis_continous=False):
        self.g_scale = g_scale
        self.vis_thresh = vis_thresh
        self.vis_continous = vis_continous
        self.gaussian = None

    def __call__(self, kp_xyc, img_size, output_size):
        kp_xyc_r = rescale_keypoints(kp_xyc, img_size, output_size)
        return self._compute_joints_gaussian_heatmaps(output_size, kp_xyc_r)

    def _compute_joints_gaussian_heatmaps(self, output_size, kp_xyc):
        w, h = output_size
        num_groups = len(joints_dict)
        group_heatmaps = np.zeros((num_groups, h, w))
        kernel = self.get_gaussian_kernel(output_size)
        g_radius = kernel.shape[0] // 2

        for group_idx, (group_name, joint_names) in enumerate(joints_dict.items()):
            temp_heatmap = np.zeros((h, w))
            for joint_name in joint_names:
                idx = keypoints_dict[joint_name]
                kp = kp_xyc[idx]
                if kp[2] <= self.vis_thresh and not self.vis_continous:
                    continue
                kpx, kpy = int(kp[0]), int(kp[1])
                if not (0 <= kpx < w and 0 <= kpy < h):
                    continue
                rt = max(0, kpy - g_radius)
                rb = min(h, kpy + g_radius + 1)
                rl = max(0, kpx - g_radius)
                rr = min(w, kpx + g_radius + 1)
                kernel_y_start = g_radius - (kpy - rt)
                kernel_y_end   = g_radius + (rb - kpy - 1)
                kernel_x_start = g_radius - (kpx - rl)
                kernel_x_end   = g_radius + (rr - kpx - 1)
                sub_kernel = kernel[kernel_y_start:kernel_y_end,
                                    kernel_x_start:kernel_x_end]
                patch = temp_heatmap[rt:rb, rl:rr]
                if patch.shape == sub_kernel.shape:
                    temp_heatmap[rt:rb, rl:rr] = np.maximum(patch, sub_kernel)
            group_heatmaps[group_idx] = temp_heatmap
        return group_heatmaps

    def get_gaussian_kernel(self, output_size):
        if self.gaussian is None:
            w, h = output_size
            g_radius = int(w / self.g_scale)
            kernel_size = g_radius * 2 + 1
            kernel = gkern(kernel_size)
            kernel = kernel / np.max(kernel)
            self.gaussian = kernel
        return self.gaussian


# ═══════════════════════════════════════════════════════════════════════
# MMPose keypoint extraction  →  Gaussian heatmap pipeline
# ═══════════════════════════════════════════════════════════════════════
def main(dataset_path, device='cpu'):
    from mmpose.apis import init_model, inference_topdown
    import mmpose

    sequences_dir = os.path.join(dataset_path, 'sequences')
    heatmap_root  = os.path.join(dataset_path, 'heatmap')

    if not os.path.isdir(sequences_dir):
        raise RuntimeError(f"sequences/ directory not found at {sequences_dir}")

    # ── Initialize MMPose RTMPose-L ──
    pkg_dir = os.path.join(os.path.dirname(mmpose.__file__), '.mim')
    config = os.path.join(
        pkg_dir,
        'configs/body_2d_keypoint/rtmpose/coco/'
        'rtmpose-l_8xb256-420e_coco-256x192.py')
    checkpoint = (
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
        'rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth')

    print(f"Loading RTMPose-L on {device}...")
    model = init_model(config, checkpoint, device=device)
    print("Model loaded.")

    # ── Heatmap converter (paper's Gaussian method) ──
    kp2mask = KeypointsToMasks(g_scale=11, vis_thresh=0.1, vis_continous=False)

    for cam_name in ['cam1', 'cam2']:
        cam_dir = os.path.join(sequences_dir, cam_name)
        if not os.path.isdir(cam_dir):
            print(f"Warning: {cam_dir} not found, skipping")
            continue

        person_dirs = sorted(os.listdir(cam_dir))
        print(f"\n{'='*60}")
        print(f"Processing {cam_name}: {len(person_dirs)} person directories")
        print(f"{'='*60}")

        cam_total = 0
        cam_detected = 0
        cam_no_kp = 0

        for p_idx, person_id in enumerate(person_dirs):
            person_img_dir = os.path.join(cam_dir, person_id)
            if not os.path.isdir(person_img_dir):
                continue

            person_heatmap_dir = os.path.join(heatmap_root, cam_name, person_id)
            os.makedirs(person_heatmap_dir, exist_ok=True)

            img_files = sorted([
                f for f in os.listdir(person_img_dir)
                if f.lower().endswith(('.png', '.jpg'))
            ])

            person_detected = 0
            person_no_kp = 0

            for img_file in img_files:
                npy_name = os.path.splitext(img_file)[0] + '.npy'
                npy_path = os.path.join(person_heatmap_dir, npy_name)
                if os.path.exists(npy_path):
                    continue

                img_path = os.path.join(person_img_dir, img_file)

                try:
                    img = Image.open(img_path).convert("RGB")
                    w, h = img.size

                    # Whole-image bbox (already a person crop)
                    bbox = np.array([[0, 0, w, h]], dtype=np.float32)
                    results = inference_topdown(model, img_path, bboxes=bbox)

                    if (results
                            and len(results[0].pred_instances.keypoints) > 0):
                        kps_xy = results[0].pred_instances.keypoints[0]       # (17,2)
                        kps_sc = results[0].pred_instances.keypoint_scores[0]  # (17,)
                        kp_array = np.column_stack(
                            [kps_xy, kps_sc]).astype(np.float32)              # (17,3)

                        above = int((kps_sc > 0.1).sum())
                        if above > 0:
                            person_detected += 1
                        else:
                            person_no_kp += 1
                    else:
                        kp_array = np.zeros((17, 3), dtype=np.float32)
                        person_no_kp += 1

                    # ── KEY CHANGE: use Gaussian blob heatmaps ──
                    heatmap = kp2mask(kp_array,
                                      img_size=(w, h),
                                      output_size=(w, h))
                    np.save(npy_path, heatmap.astype(np.float32))

                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
                    # fallback: zeros
                    heatmap = np.zeros((6, h, w), dtype=np.float32)
                    np.save(npy_path, heatmap)
                    person_no_kp += 1

            cam_total += person_detected + person_no_kp
            cam_detected += person_detected
            cam_no_kp += person_no_kp

            if (p_idx + 1) % 25 == 0 or (p_idx + 1) == len(person_dirs):
                print(f"  [{cam_name}] {p_idx+1}/{len(person_dirs)} persons done")

        print(f"\n[{cam_name}] SUMMARY: {cam_detected}/{cam_total} images "
              f"with keypoints above threshold")

    print(f"\nAll heatmaps saved to: {heatmap_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate Gaussian-blob body-part heatmaps for iLIDS-VID "
                    "(MMPose extraction + original paper rendering)")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to iLIDSVID dataset root (containing sequences/)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    args = parser.parse_args()
    main(args.dataset_path, args.device)
