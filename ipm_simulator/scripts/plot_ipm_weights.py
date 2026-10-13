#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_weight_maps(folder: Path, tag: str):
    maps = []
    for cam in range(4):
        path = folder / f"weight_cam{cam}_{tag}.png"
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Missing {path}")
        maps.append(img.astype(np.float32) / 65535.0)

    fused = cv2.imread(str(folder / f"ipm_debug_{tag}.png"), cv2.IMREAD_COLOR)
    if fused is None:
        raise FileNotFoundError(
            f"Missing fused IPM image ipm_debug_{tag}.png in {folder}"
        )

    dominant = cv2.imread(
        str(folder / f"dominant_{tag}.png"), cv2.IMREAD_UNCHANGED
    )

    return maps, fused, dominant


def plot_maps(maps, fused_bgr, tag: str, out_dir: Path):
    titles = [
        "Front weight",
        "Rear weight",
        "Left weight",
        "Right weight",
        "Fused IPM",
        "Weight sum (top-2)",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for i, ax in enumerate(axes.flat[:4]):
        im = ax.imshow(maps[i], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(titles[i])
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes.flat[4].imshow(cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB))
    axes.flat[4].set_title(titles[4])
    axes.flat[4].axis("off")

    combined = np.sum(np.stack(maps, axis=-1), axis=-1)
    axes.flat[5].imshow(combined, cmap="magma", vmin=0.0, vmax=1.0)
    axes.flat[5].set_title(titles[5])
    axes.flat[5].axis("off")

    fig.suptitle(f"IPM Seam Weights ({tag})", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"weight_heatmap_{tag}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


def plot_comparison(maps_a, fused_a_bgr, tag_a: str,
                    maps_b, fused_b_bgr, tag_b: str, out_dir: Path):
    titles = [
        "Front weight",
        "Rear weight",
        "Left weight",
        "Right weight",
        "Fused IPM",
        "Weight sum (top-2)",
    ]
    fig, axes = plt.subplots(3, 6, figsize=(18, 12))

    fused_a_rgb = cv2.cvtColor(fused_a_bgr, cv2.COLOR_BGR2RGB)
    fused_b_rgb = cv2.cvtColor(fused_b_bgr, cv2.COLOR_BGR2RGB)

    sum_a = np.sum(np.stack(maps_a, axis=-1), axis=-1)
    sum_b = np.sum(np.stack(maps_b, axis=-1), axis=-1)

    diff_maps = [np.abs(m2 - m1) for m1, m2 in zip(maps_a, maps_b)]
    diff_sum = np.abs(sum_b - sum_a)
    fused_diff = cv2.absdiff(fused_a_bgr, fused_b_bgr)
    fused_diff_gray = cv2.cvtColor(fused_diff, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if fused_diff_gray.max() > 0:
        fused_diff_vis = fused_diff_gray / fused_diff_gray.max()
    else:
        fused_diff_vis = fused_diff_gray

    rows = [
        (tag_a, maps_a, fused_a_rgb, sum_a, False),
        (tag_b, maps_b, fused_b_rgb, sum_b, False),
        (f"|{tag_b}-{tag_a}|", diff_maps, fused_diff_vis, diff_sum, True),
    ]

    for row_idx, (label, maps, fused_rgb, weight_sum, is_diff) in enumerate(rows):
        for col in range(4):
            ax = axes[row_idx, col]
            if is_diff:
                vmax = max(1e-6, np.max(maps[col]))
                im = ax.imshow(maps[col], cmap="magma", vmin=0.0, vmax=vmax)
            else:
                im = ax.imshow(maps[col], cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(titles[col])
            ax.axis("off")
            if not is_diff and row_idx == 0:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if is_diff and row_idx == 2 and col == 0:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fused_ax = axes[row_idx, 4]
        if is_diff:
            im_fused = fused_ax.imshow(fused_rgb, cmap="magma", vmin=0.0, vmax=1.0)
            fig.colorbar(im_fused, ax=fused_ax, fraction=0.046, pad=0.04)
        else:
            fused_ax.imshow(fused_rgb)
        fused_ax.set_title(titles[4])
        fused_ax.axis("off")

        sum_ax = axes[row_idx, 5]
        if is_diff:
            vmax = max(1e-6, np.max(weight_sum))
            im_sum = sum_ax.imshow(weight_sum, cmap="magma", vmin=0.0, vmax=vmax)
            fig.colorbar(im_sum, ax=sum_ax, fraction=0.046, pad=0.04)
        else:
            im_sum = sum_ax.imshow(weight_sum, cmap="magma", vmin=0.0, vmax=1.0)
            if row_idx == 0:
                fig.colorbar(im_sum, ax=sum_ax, fraction=0.046, pad=0.04)
        sum_ax.set_title(titles[5])
        sum_ax.axis("off")

        axes[row_idx, 0].set_ylabel(label, fontsize=12)

    fig.suptitle(f"IPM Seam Weights Comparison: {tag_a} vs {tag_b}",
                 fontsize=16, y=0.97)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"weight_heatmap_{tag_a}_vs_{tag_b}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot IPM seam weight heatmaps."
    )
    parser.add_argument("--folder", type=Path, default=Path("output"))
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument(
        "--compare-tag", type=str, default=None,
        help="Optional second tag to produce a side-by-side comparison figure."
    )
    args = parser.parse_args()

    maps, fused, _ = load_weight_maps(args.folder, args.tag)
    if args.compare_tag:
        maps_b, fused_b, _ = load_weight_maps(args.folder, args.compare_tag)
        plot_comparison(maps, fused, args.tag, maps_b, fused_b,
                        args.compare_tag, args.out)
    else:
        plot_maps(maps, fused, args.tag, args.out)


if __name__ == "__main__":
    main()
