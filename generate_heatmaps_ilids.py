"""
generate_heatmaps_ilids.py

Generate 6-channel body-part heatmaps for iLIDS-VID using MMPose (RTMPose).
Run in the 'mmpose' conda env.

Usage:
    python generate_heatmaps_ilids.py --dataset_path ./data/iLIDSVID
    
    # With GPU:
    python generate_heatmaps_ilids.py --dataset_path ./data/iLIDSVID --device cuda

Output:
    ./data/iLIDSVID/heatmap/cam1/person001/cam1_person001_00317.npy  (shape: 6, H, W)
    ...
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter

# ─── COCO keypoint indices ───
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LS, RS, LE, RE, LW, RW = 5, 6, 7, 8, 9, 10
LH, RH, LK, RK, LA, RA = 11, 12, 13, 14, 15, 16


def keypoints_to_heatmap(kp_array, img_w, img_h, vis_thresh=0.1):
    """
    Convert 17 COCO keypoints into 6-channel body-part heatmaps using
    region filling (head: convex hull, torso: filled quad, limbs: thick
    line segments) with Gaussian-blurred edges and L/R sanity correction.

    Args:
        kp_array: np.array shape (17, 3) — [x, y, confidence]
        img_w, img_h: original image dimensions
    Returns:
        np.array shape (6, img_h, img_w)
    """
    heatmaps = np.zeros((6, img_h, img_w))

    blur_sigma = max(1.0, img_w / 16.0)
    line_thickness = max(1, int(img_w / 8))

    def vis(i):
        return kp_array[i, 2] > vis_thresh

    def pt(i):
        return (int(kp_array[i, 0]), int(kp_array[i, 1]))

    def in_bounds(i):
        x, y = int(kp_array[i, 0]), int(kp_array[i, 1])
        return 0 <= x < img_w and 0 <= y < img_h

    def usable(i):
        return vis(i) and in_bounds(i)

    # --- Channel 0: Head (convex hull of visible keypoints) ---
    head_pts = [pt(i) for i in [NOSE, LEYE, REYE, LEAR, REAR] if usable(i)]
    if len(head_pts) >= 3:
        hull = cv2.convexHull(np.array(head_pts, dtype=np.int32))
        cv2.fillConvexPoly(heatmaps[0], hull, 1.0)
    elif head_pts:
        for p in head_pts:
            cv2.circle(heatmaps[0], p, line_thickness, 1.0, -1)

    # --- Channel 1: Torso (filled quadrilateral LS→RS→RH→LH) ---
    torso_order = [LS, RS, RH, LH]
    torso_usable = [i for i in torso_order if usable(i)]
    if len(torso_usable) == 4:
        pts = np.array([pt(i) for i in torso_order], dtype=np.int32)
        cv2.fillPoly(heatmaps[1], [pts], 1.0)
    elif len(torso_usable) >= 3:
        hull = cv2.convexHull(np.array([pt(i) for i in torso_usable], dtype=np.int32))
        cv2.fillConvexPoly(heatmaps[1], hull, 1.0)
    elif len(torso_usable) == 2:
        pts = [pt(i) for i in torso_usable]
        cv2.line(heatmaps[1], pts[0], pts[1], 1.0, line_thickness)
    elif len(torso_usable) == 1:
        cv2.circle(heatmaps[1], pt(torso_usable[0]), line_thickness, 1.0, -1)

    # --- Channels 2-5: Limbs (thick line segments) ---
    limbs = [
        (2, LE, LW),  # left arm
        (3, RE, RW),  # right arm
        (4, LK, LA),  # left leg
        (5, RK, RA),  # right leg
    ]
    for ch, j1, j2 in limbs:
        u1, u2 = usable(j1), usable(j2)
        if u1 and u2:
            cv2.line(heatmaps[ch], pt(j1), pt(j2), 1.0, line_thickness)
        elif u1:
            cv2.circle(heatmaps[ch], pt(j1), line_thickness, 1.0, -1)
        elif u2:
            cv2.circle(heatmaps[ch], pt(j2), line_thickness, 1.0, -1)

    # --- Gaussian blur for soft edges ---
    for i in range(6):
        if heatmaps[i].max() > 0:
            heatmaps[i] = gaussian_filter(heatmaps[i], sigma=blur_sigma)

    # --- L/R sanity check based on torso orientation ---
    if usable(LS) and usable(RS):
        shoulder_gap = abs(kp_array[LS, 0] - kp_array[RS, 0])
        if shoulder_gap > img_w * 0.05:  # skip if near-profile view
            # In COCO, "left" = person's left. When facing camera,
            # person's left appears at higher x (image-right).
            facing_camera = kp_array[LS, 0] > kp_array[RS, 0]

            # Swap arms if L/R placement contradicts torso orientation
            if usable(LE) and usable(RE):
                left_to_image_right = kp_array[LE, 0] > kp_array[RE, 0]
                if facing_camera != left_to_image_right:
                    heatmaps[2], heatmaps[3] = heatmaps[3].copy(), heatmaps[2].copy()

            # Swap legs if L/R placement contradicts torso orientation
            if usable(LK) and usable(RK):
                left_to_image_right = kp_array[LK, 0] > kp_array[RK, 0]
                if facing_camera != left_to_image_right:
                    heatmaps[4], heatmaps[5] = heatmaps[5].copy(), heatmaps[4].copy()

    # --- Normalize each channel to [0, 1] ---
    for i in range(6):
        mx = heatmaps[i].max()
        if mx > 0:
            heatmaps[i] /= mx

    return heatmaps


# ─── Main pipeline ───
def main(dataset_path, device='cpu'):
    from mmpose.apis import init_model, inference_topdown
    import mmpose

    sequences_dir = os.path.join(dataset_path, 'sequences')
    heatmap_root = os.path.join(dataset_path, 'heatmap')

    if not os.path.isdir(sequences_dir):
        raise RuntimeError(f"sequences/ directory not found at {sequences_dir}")

    # Initialize MMPose RTMPose model
    pkg_dir = os.path.join(os.path.dirname(mmpose.__file__), '.mim')
    config = os.path.join(pkg_dir,
        'configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py')
    checkpoint = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
                  'rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth')

    print(f"Loading RTMPose-L on {device}...")
    model = init_model(config, checkpoint, device=device)
    print("Model loaded.")

    for cam_name in ['cam1', 'cam2']:
        cam_dir = os.path.join(sequences_dir, cam_name)
        if not os.path.isdir(cam_dir):
            print(f"Warning: {cam_dir} not found, skipping")
            continue

        person_dirs = sorted(os.listdir(cam_dir))
        print(f"\n{'='*60}")
        print(f"Processing {cam_name}: {len(person_dirs)} person directories")
        print(f"{'='*60}")

        cam_total_detected = 0
        cam_total_no_detection = 0

        for p_idx, person_id in enumerate(person_dirs):
            person_img_dir = os.path.join(cam_dir, person_id)
            if not os.path.isdir(person_img_dir):
                continue

            person_heatmap_dir = os.path.join(heatmap_root, cam_name, person_id)
            os.makedirs(person_heatmap_dir, exist_ok=True)

            img_files = sorted([f for f in os.listdir(person_img_dir)
                               if f.lower().endswith(('.png', '.jpg'))])

            person_detected = 0
            person_no_detection = 0
            first_img_logged = False

            for img_file in img_files:
                npy_name = os.path.splitext(img_file)[0] + '.npy'
                npy_path = os.path.join(person_heatmap_dir, npy_name)
                if os.path.exists(npy_path):
                    continue

                img_path = os.path.join(person_img_dir, img_file)

                try:
                    img = Image.open(img_path).convert("RGB")
                    w, h = img.size

                    # Use whole image as bounding box (it's already a person crop)
                    bbox = np.array([[0, 0, w, h]], dtype=np.float32)
                    results = inference_topdown(model, img_path, bboxes=bbox)

                    if results and len(results[0].pred_instances.keypoints) > 0:
                        kps_xy = results[0].pred_instances.keypoints[0]           # (17, 2)
                        kps_scores = results[0].pred_instances.keypoint_scores[0]  # (17,)
                        # Build (17, 3) array matching COCO format
                        kp_array = np.column_stack([kps_xy, kps_scores]).astype(np.float32)

                        above_thresh = int((kps_scores > 0.1).sum())
                        if above_thresh > 0:
                            person_detected += 1
                        else:
                            person_no_detection += 1

                        # Log score details for the first image of each person
                        if not first_img_logged:
                            print(f"    [{cam_name}/{person_id}] first image: {img_file} | "
                                  f"score min={kps_scores.min():.3f} max={kps_scores.max():.3f} "
                                  f"mean={kps_scores.mean():.3f} | "
                                  f"kps above thresh (>0.1): {above_thresh}/17")
                            first_img_logged = True
                    else:
                        kp_array = np.zeros((17, 3), dtype=np.float32)
                        person_no_detection += 1
                        if not first_img_logged:
                            print(f"    [{cam_name}/{person_id}] first image: {img_file} | NO results returned by model")
                            first_img_logged = True

                    heatmap = keypoints_to_heatmap(kp_array, w, h)
                    np.save(npy_path, heatmap)

                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
                    heatmap = np.zeros((6, h, w), dtype=np.float32)
                    np.save(npy_path, heatmap)
                    person_no_detection += 1

            cam_total_detected += person_detected
            cam_total_no_detection += person_no_detection
            person_total = person_detected + person_no_detection
            print(f"  [{cam_name}] {person_id}: {person_detected}/{person_total} images with detections, "
                  f"{person_no_detection} with no keypoints above threshold")

        print(f"\n[{cam_name}] SUMMARY: {cam_total_detected} images with detections, "
              f"{cam_total_no_detection} images with no detections "
              f"({cam_total_detected + cam_total_no_detection} total processed)")

    print(f"\nAll heatmaps saved to: {heatmap_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate 6-channel body-part heatmaps for iLIDS-VID using MMPose RTMPose")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to iLIDSVID dataset root (containing sequences/)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    args = parser.parse_args()
    main(args.dataset_path, args.device)