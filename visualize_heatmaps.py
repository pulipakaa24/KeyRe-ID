"""
visualize_heatmaps.py — iLIDS-VID heatmap overlay viewer

Run in the 'heatmap' conda environment:
    conda run -n heatmap python visualize_heatmaps.py [options]

Or activate the env first:
    conda activate heatmap
    python visualize_heatmaps.py --dataset_root /path/to/iLIDS-VID

Controls (interactive mode):
    ← / →    previous / next frame
    ↑ / ↓    previous / next person
    1–6      toggle individual body-part channel
    a        toggle all-channels combined view
    +/-      increase / decrease heatmap alpha
    s        save current figure to PNG
    q        quit

Options:
    --dataset_root  path to the iLIDS-VID dataset root directory (required)
    --cam           cam1 or cam2 (default: cam1)
    --person        e.g. person001 (default: first available)
    --frame         frame filename stem, e.g. cam1_person001_00317 (default: first)
    --alpha         initial overlay opacity 0-1 (default: 0.5)
    --save_dir      if set, save all frames for chosen person instead of interactive
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image

# ── Body-part channel names & colours ─────────────────────────────────────────
CHANNEL_NAMES = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
CHANNEL_CMAPS = ["Reds", "Blues", "Greens", "Oranges", "Purples", "YlOrBr"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_persons(cam, sequences_root):
    cam_dir = os.path.join(sequences_root, cam)
    if not os.path.isdir(cam_dir):
        sys.exit(f"[ERROR] Camera directory not found: {cam_dir}")
    persons = sorted([
        d for d in os.listdir(cam_dir)
        if os.path.isdir(os.path.join(cam_dir, d))
    ])
    if not persons:
        sys.exit(f"[ERROR] No person directories found under {cam_dir}")
    return persons


def list_frames(cam, person, sequences_root):
    img_dir = os.path.join(sequences_root, cam, person)
    frames = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(".png") and not f.endswith(".Identifier")
    ])
    return frames


def load_image(cam, person, frame_file, sequences_root):
    path = os.path.join(sequences_root, cam, person, frame_file)
    return np.array(Image.open(path).convert("RGB"))


def load_heatmap(cam, person, frame_file, heatmap_root):
    """Load (6, H, W) heatmap for the given frame. Returns zeros if missing."""
    stem = os.path.splitext(frame_file)[0]
    npy_path = os.path.join(heatmap_root, cam, person, stem + ".npy")
    if not os.path.exists(npy_path):
        return None, npy_path
    heatmap = np.load(npy_path).astype(np.float64)  # (6, H, W)
    return heatmap, npy_path


def normalise_channel(channel):
    """Min-max normalise a 2-D array to [0, 1]."""
    mn, mx = channel.min(), channel.max()
    if mx - mn < 1e-8:
        return np.zeros_like(channel)
    return (channel - mn) / (mx - mn)


def overlay_heatmap_on_image(img_rgb, heatmap, channels_on, alpha):
    """
    Blend selected heatmap channels onto img_rgb.
    Returns an RGBA float image in [0, 1].
    """
    img_f = img_rgb.astype(np.float64) / 255.0
    H, W = img_f.shape[:2]
    blended = img_f.copy()

    for ch_idx in channels_on:
        if ch_idx >= heatmap.shape[0]:
            continue
        ch = normalise_channel(heatmap[ch_idx])           # (H_h, W_h)
        # Resize heatmap channel to image size if needed
        if ch.shape != (H, W):
            from PIL import Image as PILImage
            ch_img = PILImage.fromarray((ch * 255).astype(np.uint8))
            ch_img = ch_img.resize((W, H), PILImage.BILINEAR)
            ch = np.array(ch_img).astype(np.float64) / 255.0

        cmap = plt.get_cmap(CHANNEL_CMAPS[ch_idx])
        colour = cmap(ch)[..., :3]                         # (H, W, 3)
        mask = ch[..., np.newaxis]                         # (H, W, 1)
        blended = blended * (1 - alpha * mask) + colour * alpha * mask

    return np.clip(blended, 0, 1)


# ── Batch-save mode ───────────────────────────────────────────────────────────

def batch_save(cam, person, save_dir, alpha, sequences_root, heatmap_root):
    os.makedirs(save_dir, exist_ok=True)
    frames = list_frames(cam, person, sequences_root)
    if not frames:
        sys.exit(f"[ERROR] No frames found for {cam}/{person}")

    channels_on = list(range(6))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    plt.tight_layout(pad=2)

    for frame_file in frames:
        img = load_image(cam, person, frame_file, sequences_root)
        heatmap, npy_path = load_heatmap(cam, person, frame_file, heatmap_root)

        if heatmap is None:
            print(f"  [WARN] Heatmap not found: {npy_path}")
            heatmap = np.zeros((6, img.shape[0], img.shape[1]))

        _render_figure(fig, axes, img, heatmap, channels_on, alpha,
                       cam, person, frame_file)

        stem = os.path.splitext(frame_file)[0]
        out_path = os.path.join(save_dir, stem + "_overlay.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)
    print(f"\nDone. {len(frames)} frames saved to {save_dir}")


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render_figure(fig, axes, img, heatmap, channels_on, alpha,
                   cam, person, frame_file):
    """Draw all 8 panels onto an existing figure."""
    for ax in axes.flat:
        ax.cla()
        ax.axis("off")

    # Panel 0: original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original", fontsize=9)

    # Panel 1: all-channels combined overlay
    combo = overlay_heatmap_on_image(img, heatmap, list(range(6)), alpha)
    axes[0, 1].imshow(combo)
    axes[0, 1].set_title("All channels", fontsize=9)

    # Panels 2–7: one per body-part channel
    panel_positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    for ch_idx, (r, c) in enumerate(panel_positions):
        ax = axes[r][c]
        active = ch_idx in channels_on
        if active:
            overlay = overlay_heatmap_on_image(img, heatmap, [ch_idx], alpha)
            ax.imshow(overlay)
        else:
            ax.imshow(img)
        title = CHANNEL_NAMES[ch_idx]
        colour = "green" if active else "gray"
        ax.set_title(title, fontsize=9, color=colour)

        # Draw per-channel heatmap stats in corner
        ch = heatmap[ch_idx]
        ax.text(2, img.shape[0] - 4,
                f"max={ch.max():.3f}",
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

    fig.suptitle(
        f"{cam} / {person} / {frame_file}   (alpha={alpha:.2f})",
        fontsize=11, y=1.01
    )
    fig.canvas.draw_idle()


# ── Interactive mode ──────────────────────────────────────────────────────────

class HeatmapViewer:
    def __init__(self, cam, start_person, start_frame, alpha,
                 sequences_root, heatmap_root):
        self.cam = cam
        self.sequences_root = sequences_root
        self.heatmap_root = heatmap_root
        self.persons = list_persons(cam, sequences_root)
        self.alpha = alpha
        self.channels_on = set(range(6))   # all on by default

        # Resolve starting person index
        if start_person in self.persons:
            self.p_idx = self.persons.index(start_person)
        else:
            self.p_idx = 0

        # Load frame list for starting person
        self.frames = list_frames(self.cam, self.persons[self.p_idx],
                                  self.sequences_root)
        self.f_idx = 0
        if start_frame:
            matches = [i for i, f in enumerate(self.frames)
                       if os.path.splitext(f)[0] == start_frame]
            if matches:
                self.f_idx = matches[0]

        self.fig, self.axes = plt.subplots(2, 4, figsize=(18, 8))
        plt.subplots_adjust(left=0.02, right=0.98, top=0.93,
                            bottom=0.05, wspace=0.05, hspace=0.25)
        self._add_legend()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    def _add_legend(self):
        legend_text = (
            "← → : prev/next frame   ↑ ↓ : prev/next person\n"
            "1-6 : toggle channel   a : toggle all   +/- : alpha   s : save   q : quit"
        )
        self.fig.text(0.5, 0.005, legend_text,
                      ha="center", va="bottom", fontsize=8,
                      color="gray")

    def _draw(self):
        person = self.persons[self.p_idx]
        frame_file = self.frames[self.f_idx]
        img = load_image(self.cam, person, frame_file, self.sequences_root)
        heatmap, npy_path = load_heatmap(self.cam, person, frame_file,
                                         self.heatmap_root)

        if heatmap is None:
            print(f"[WARN] Heatmap not found: {npy_path}")
            heatmap = np.zeros((6, img.shape[0], img.shape[1]))

        _render_figure(
            self.fig, self.axes,
            img, heatmap,
            sorted(self.channels_on),
            self.alpha,
            self.cam, person, frame_file,
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        key = event.key
        if key == "right":
            self.f_idx = (self.f_idx + 1) % len(self.frames)
        elif key == "left":
            self.f_idx = (self.f_idx - 1) % len(self.frames)
        elif key == "up":
            self.p_idx = (self.p_idx - 1) % len(self.persons)
            self.frames = list_frames(self.cam, self.persons[self.p_idx],
                                      self.sequences_root)
            self.f_idx = 0
        elif key == "down":
            self.p_idx = (self.p_idx + 1) % len(self.persons)
            self.frames = list_frames(self.cam, self.persons[self.p_idx],
                                      self.sequences_root)
            self.f_idx = 0
        elif key in "123456":
            ch = int(key) - 1
            if ch in self.channels_on:
                self.channels_on.discard(ch)
            else:
                self.channels_on.add(ch)
        elif key == "a":
            if len(self.channels_on) == 6:
                self.channels_on.clear()
            else:
                self.channels_on = set(range(6))
        elif key in ("+", "="):
            self.alpha = min(1.0, self.alpha + 0.05)
        elif key == "-":
            self.alpha = max(0.0, self.alpha - 0.05)
        elif key == "s":
            self._save()
            return
        elif key == "q":
            plt.close(self.fig)
            return
        else:
            return
        self._draw()

    def _save(self):
        person = self.persons[self.p_idx]
        frame_file = self.frames[self.f_idx]
        stem = os.path.splitext(frame_file)[0]
        out = f"{stem}_overlay.png"
        self.fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    def show(self):
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", required=True,
                   help="Path to the iLIDS-VID dataset root directory")
    p.add_argument("--cam", default="cam1", choices=["cam1", "cam2"])
    p.add_argument("--person", default=None,
                   help="e.g. person001 (default: first available)")
    p.add_argument("--frame", default=None,
                   help="Frame filename stem (without extension)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Heatmap overlay opacity (default: 0.5)")
    p.add_argument("--save_dir", default=None,
                   help="If set, batch-save all frames for the chosen person "
                        "and exit (no interactive window)")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    sequences_root = os.path.join(dataset_root, "sequences")
    heatmap_root = os.path.join(dataset_root, "heatmap")

    # Resolve default person
    if args.person is None:
        persons = list_persons(args.cam, sequences_root)
        args.person = persons[0]

    if args.save_dir:
        batch_save(args.cam, args.person, args.save_dir, args.alpha,
                   sequences_root, heatmap_root)
    else:
        viewer = HeatmapViewer(
            cam=args.cam,
            start_person=args.person,
            start_frame=args.frame,
            alpha=args.alpha,
            sequences_root=sequences_root,
            heatmap_root=heatmap_root,
        )
        viewer.show()


if __name__ == "__main__":
    main()
