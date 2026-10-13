#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------- I/O helpers ---------------------- #

PREFERRED_PCD_NAMES = [
    "cloudGlobal.pcd",
    "cloudSurf.pcd",
    "cloudCorner.pcd",
    "custom_map_scaled_shifted.pcd",
    "custom_map_scaled.pcd",
    "custom_map_cleaned.pcd",
    "custom_map.pcd",
]


def parse_pcd_ascii(path: Path) -> np.ndarray:
    """Load ASCII PCD with fields containing at least x y z."""
    with path.open("r") as f:
        lines = f.readlines()

    data_start = None
    fields: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "FIELDS":
            fields = {name: i for i, name in enumerate(parts[1:])}
        if parts[0] == "DATA":
            data_start = idx + 1
            break

    if data_start is None:
        raise ValueError(f"No DATA section found in PCD: {path}")
    if not {"x", "y", "z"}.issubset(fields.keys()):
        raise ValueError(f"PCD missing xyz fields: {path}")

    numeric_lines = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("DATA"):
            continue
        numeric_lines.append(stripped)

    if not numeric_lines:
        raise ValueError(f"No point data lines found in PCD: {path}")

    arr = np.loadtxt(numeric_lines)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    xyz = arr[:, [fields["x"], fields["y"], fields["z"]]]
    return xyz.astype(np.float32)


def load_kitti_bin(path: Path, max_xy_range: float) -> np.ndarray:
    """Load KITTI-style XYZI float32 binary as xyz."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Unexpected KITTI bin length (not divisible by 4): {path}")
    pts = raw.reshape(-1, 4)[:, :3]
    mask = np.isfinite(pts).all(axis=1)
    if max_xy_range > 0:
        mask &= (np.abs(pts[:, 0]) <= max_xy_range) & (np.abs(pts[:, 1]) <= max_xy_range)
    return pts[mask]


def resolve_cloud_path(path_str: str) -> Path:
    """Resolve a file or directory to a concrete point cloud file."""
    p = Path(path_str)
    if p.is_file():
        return p
    if p.is_dir():
        for name in PREFERRED_PCD_NAMES:
            cand = p / name
            if cand.is_file():
                return cand
        # fallback: first .pcd
        for cand in p.iterdir():
            if cand.is_file() and cand.suffix.lower() == ".pcd":
                return cand
    raise FileNotFoundError(f"Point cloud not found for path: {path_str}")


def load_point_cloud(path_str: str, max_xy_range: float, max_z_range: float) -> Tuple[np.ndarray, str]:
    """Load point cloud as Nx3 float32 and return base name for outputs."""
    resolved = resolve_cloud_path(path_str)
    base = resolved.stem if resolved.is_file() else resolved.name
    if resolved.suffix.lower() == ".pcd":
        pts = parse_pcd_ascii(resolved)
    elif resolved.suffix.lower() == ".bin":
        pts = load_kitti_bin(resolved, max_xy_range)
    else:
        raise ValueError(f"Unsupported point cloud type: {resolved}")
    mask = np.isfinite(pts).all(axis=1)
    if max_xy_range > 0:
        mask &= (np.abs(pts[:, 0]) <= max_xy_range) & (np.abs(pts[:, 1]) <= max_xy_range)
    if max_z_range > 0:
        mask &= np.abs(pts[:, 2]) <= max_z_range
    pts = pts[mask]
    if pts.size == 0:
        raise ValueError(f"No valid points loaded from {resolved}")
    return pts, base


# ---------------------- Metric computation ---------------------- #

def compute_extent(points: np.ndarray, padding: float) -> Tuple[float, float, float, float]:
    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    min_x, min_y = min_xy - padding
    max_x, max_y = max_xy + padding
    return float(min_x), float(max_x), float(min_y), float(max_y)


def build_grid(points: np.ndarray, resolution: float, padding: float):
    min_x, max_x, min_y, max_y = compute_extent(points, padding)
    width = max_x - min_x
    height = max_y - min_y
    cols = int(math.ceil(width / resolution))
    rows = int(math.ceil(height / resolution))
    return (min_x, max_x, min_y, max_y), rows, cols


def bin_points(points: np.ndarray, extent, rows: int, cols: int, resolution: float, split_ratio: float):
    min_x, max_x, min_y, max_y = extent
    max_y_val = max_y  # for row computation

    # Map to grid indices
    cols_idx = np.floor((points[:, 0] - min_x) / resolution).astype(np.int32)
    rows_idx = np.floor((max_y_val - points[:, 1]) / resolution).astype(np.int32)

    valid = (cols_idx >= 0) & (cols_idx < cols) & (rows_idx >= 0) & (rows_idx < rows)
    cols_idx = cols_idx[valid]
    rows_idx = rows_idx[valid]
    zs = points[valid, 2].astype(np.float32)

    counts = np.zeros((rows, cols), dtype=np.float32)
    sum_z = np.zeros_like(counts)
    sumsq_z = np.zeros_like(counts)
    counts_a = np.zeros_like(counts)
    counts_b = np.zeros_like(counts)

    split_index = int(split_ratio * cols_idx.size)
    mask_a = np.zeros(cols_idx.size, dtype=bool)
    mask_a[:split_index] = True

    np.add.at(counts, (rows_idx, cols_idx), 1.0)
    np.add.at(sum_z, (rows_idx, cols_idx), zs)
    np.add.at(sumsq_z, (rows_idx, cols_idx), zs * zs)

    np.add.at(counts_a, (rows_idx[mask_a], cols_idx[mask_a]), 1.0)
    np.add.at(counts_b, (rows_idx[~mask_a], cols_idx[~mask_a]), 1.0)

    return counts, sum_z, sumsq_z, counts_a, counts_b


def occupancy_probability(counts: np.ndarray) -> np.ndarray:
    max_v = counts.max()
    if max_v <= 0:
        return np.zeros_like(counts, dtype=np.float32)
    return counts.astype(np.float32) / max_v


def overlap_heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.where(mx > 0, mn / mx, 0.0)
    return overlap.astype(np.float32)


def noise_std(counts: np.ndarray, sum_z: np.ndarray, sumsq_z: np.ndarray) -> np.ndarray:
    mean = np.where(counts > 0, sum_z / np.maximum(counts, 1e-6), 0.0)
    mean_sq = np.where(counts > 0, sumsq_z / np.maximum(counts, 1e-6), 0.0)
    var = np.maximum(0.0, mean_sq - mean * mean)
    std = np.where(counts >= 2.0, np.sqrt(var), 0.0)
    return std.astype(np.float32)


# ---------------------- Visualization helpers ---------------------- #

def to_heatmap(values: np.ndarray, log_scale: bool, colormap: int) -> np.ndarray:
    if log_scale:
        values = np.log1p(values)
    max_v = values.max()
    if max_v <= 0:
        max_v = 1.0
    norm = (values / max_v * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, colormap)
    color[values <= 0] = 0
    return color


def save_yaml(path: Path, extent, resolution: float, label: str, mat: np.ndarray):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open {path} for writing")
    min_x, max_x, min_y, max_y = extent
    fs.write("label", label)
    fs.write("resolution_m", resolution)
    fs.write("extent_min_x", float(min_x))
    fs.write("extent_max_x", float(max_x))
    fs.write("extent_min_y", float(min_y))
    fs.write("extent_max_y", float(max_y))
    fs.write("values", mat)
    fs.release()


def save_outputs(base: str, out_dir: Path, extent, resolution: float, counts, sum_z, sumsq_z, counts_a, counts_b):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Occupancy / free-space confidence
    occ = occupancy_probability(counts)
    occ_img = to_heatmap(occ, log_scale=False, colormap=cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out_dir / f"{base}_occupancy_confidence.png"), occ_img)
    save_yaml(out_dir / f"{base}_occupancy_confidence.yml", extent, resolution, "occupancy_probability", occ)

    # Map consistency / overlap
    overlap = overlap_heatmap(counts_a, counts_b)
    overlap_img = to_heatmap(overlap, log_scale=False, colormap=cv2.COLORMAP_MAGMA)
    cv2.imwrite(str(out_dir / f"{base}_overlap_heatmap.png"), overlap_img)
    save_yaml(out_dir / f"{base}_overlap_heatmap.yml", extent, resolution, "overlap_ratio", overlap)

    # Smoothness / noise (std of z)
    noise = noise_std(counts, sum_z, sumsq_z)
    noise_img = to_heatmap(noise, log_scale=True, colormap=cv2.COLORMAP_PLASMA)
    cv2.imwrite(str(out_dir / f"{base}_noise_heatmap.png"), noise_img)
    save_yaml(out_dir / f"{base}_noise_heatmap.yml", extent, resolution, "noise_std_z", noise)


# ---------------------- CLI ---------------------- #

def main():
    parser = argparse.ArgumentParser(description="Render map indicators (occupancy, overlap, noise) from point clouds.")
    parser.add_argument("--point_cloud", required=True, help="Path to .pcd/.bin or directory containing a point cloud")
    parser.add_argument("--output_dir", default="output/Quantitative_indicators/new_indicators", help="Output directory")
    parser.add_argument("--resolution", type=float, default=0.2, help="Grid resolution in meters")
    parser.add_argument("--padding", type=float, default=1.0, help="Padding around map extent in meters")
    parser.add_argument("--overlap_split_ratio", type=float, default=0.5, help="Split ratio (0-1) for overlap computation")
    parser.add_argument("--max_xy_range", type=float, default=500.0, help="Max |x|,|y| to keep (<=0 disables)")
    parser.add_argument("--max_z_range", type=float, default=50.0, help="Max |z| to keep (<=0 disables)")
    args = parser.parse_args()

    if not (0.0 < args.overlap_split_ratio < 1.0):
        parser.error("--overlap_split_ratio must be in (0,1)")
    if args.resolution <= 0.0:
        parser.error("--resolution must be positive")

    points, base = load_point_cloud(args.point_cloud, args.max_xy_range, args.max_z_range)
    extent, rows, cols = build_grid(points, args.resolution, args.padding)
    counts, sum_z, sumsq_z, counts_a, counts_b = bin_points(
        points, extent, rows, cols, args.resolution, args.overlap_split_ratio
    )

    out_dir = Path(args.output_dir)
    save_outputs(base, out_dir, extent, args.resolution, counts, sum_z, sumsq_z, counts_a, counts_b)
    print(f"Saved indicators for {base} to {out_dir}")


if __name__ == "__main__":
    main()
