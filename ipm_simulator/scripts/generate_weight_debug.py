#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np

MODE_CONFIGS = {
    "mixed": {"distance": True, "image": True, "direction": True},
    "baseline": {"distance": True, "image": False, "direction": False},
    "distance_only": {"distance": True, "image": False, "direction": True},
    "uniform": {"distance": False, "image": False, "direction": False},
}


def load_camera(index: int, data_dir: Path):
    intrinsic_path = data_dir / f"{index}_intrinsic.yaml"
    extrinsic_path = data_dir / f"{index}_extrinsic.yaml"

    intrinsic_fs = cv2.FileStorage(str(intrinsic_path), cv2.FILE_STORAGE_READ)
    if not intrinsic_fs.isOpened():
        raise FileNotFoundError(f"Failed to open {intrinsic_path}")
    width = int(intrinsic_fs.getNode("image_width").real())
    height = int(intrinsic_fs.getNode("image_height").real())
    xi = intrinsic_fs.getNode("mirror_parameters").getNode("xi").real()
    gamma1 = intrinsic_fs.getNode("projection_parameters").getNode("gamma1").real()
    gamma2 = intrinsic_fs.getNode("projection_parameters").getNode("gamma2").real()
    u0 = intrinsic_fs.getNode("projection_parameters").getNode("u0").real()
    v0 = intrinsic_fs.getNode("projection_parameters").getNode("v0").real()
    intrinsic_fs.release()

    K = np.array(
        [
            [gamma1, 0.0, u0],
            [0.0, gamma2, v0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    extrinsic_fs = cv2.FileStorage(str(extrinsic_path), cv2.FILE_STORAGE_READ)
    if not extrinsic_fs.isOpened():
        raise FileNotFoundError(f"Failed to open {extrinsic_path}")
    trans_node = extrinsic_fs.getNode("transform").getNode("translation")
    rot_node = extrinsic_fs.getNode("transform").getNode("rotation")
    t = np.array(
        [
            trans_node.getNode("x").real(),
            trans_node.getNode("y").real(),
            trans_node.getNode("z").real(),
        ],
        dtype=np.float64,
    )
    q = np.array(
        [
            rot_node.getNode("w").real(),
            rot_node.getNode("x").real(),
            rot_node.getNode("y").real(),
            rot_node.getNode("z").real(),
        ],
        dtype=np.float64,
    )  # w,x,y,z
    extrinsic_fs.release()
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        R = np.eye(3, dtype=np.float64)
    else:
        w, x, y, z = q / q_norm
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    # Transform from vehicle to camera frame
    R_cam_vehicle = R.T
    t_cam_vehicle = -R_cam_vehicle @ t

    return {
        "width": width,
        "height": height,
        "xi": float(xi),
        "K": K,
        "center_x": width * 0.5,
        "center_y": height * 0.5,
        "R_cam_vehicle": R_cam_vehicle,
        "t_cam_vehicle": t_cam_vehicle,
    }


def compute_weight_maps(cameras, mode="mixed", ipm_w=1000, ipm_h=1000,
                        pixel_scale=0.02, ipm_height=0.0):
    half_h = ipm_h * 0.5
    half_w = ipm_w * 0.5

    u_coords = np.arange(ipm_w, dtype=np.float32)
    v_coords = np.arange(ipm_h, dtype=np.float32)
    U, V = np.meshgrid(u_coords, v_coords, indexing="xy")

    px_grid = -(half_h - U) * pixel_scale
    py_grid = (half_w - V) * pixel_scale
    pz_grid = np.full_like(px_grid, ipm_height, dtype=np.float32)

    weights = []
    config = MODE_CONFIGS.get(mode, MODE_CONFIGS["mixed"])
    use_distance = config["distance"]
    use_image = config["image"]
    use_direction = config["direction"]

    for idx, cam in enumerate(cameras):
        R = cam["R_cam_vehicle"]
        t = cam["t_cam_vehicle"]
        xi = cam["xi"]
        K = cam["K"]

        # Transform to camera coordinates
        pc_x = R[0, 0] * px_grid + R[0, 1] * py_grid + R[0, 2] * pz_grid + t[0]
        pc_y = R[1, 0] * px_grid + R[1, 1] * py_grid + R[1, 2] * pz_grid + t[1]
        pc_z = R[2, 0] * px_grid + R[2, 1] * py_grid + R[2, 2] * pz_grid + t[2]

        d = np.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
        valid = d > 1e-9
        valid &= pc_z >= 0.0

        denominator = d + xi * pc_z
        valid &= denominator > 0.0

        pc_norm_x = np.zeros_like(pc_x)
        pc_norm_y = np.zeros_like(pc_y)
        pc_norm_z = np.zeros_like(pc_z)
        pc_norm_x[valid] = pc_x[valid] / d[valid]
        pc_norm_y[valid] = pc_y[valid] / d[valid]
        pc_norm_z[valid] = pc_z[valid] / d[valid]

        pC_z = pc_norm_z + xi
        valid &= pC_z > 0.0

        p_unit_x = np.zeros_like(pc_x)
        p_unit_y = np.zeros_like(pc_y)
        p_unit_x[valid] = pc_norm_x[valid] / pC_z[valid]
        p_unit_y[valid] = pc_norm_y[valid] / pC_z[valid]

        uv0 = K[0, 0] * p_unit_x + K[0, 2]
        uv1 = K[1, 1] * p_unit_y + K[1, 2]

        valid &= (uv0 >= 0.0) & (uv0 < cam["width"])
        valid &= (uv1 >= 0.0) & (uv1 < cam["height"])

        dist_weight = np.ones_like(pc_x, dtype=np.float32)
        if use_distance:
            dist_weight[valid] = 1.0 / (1.0 + d[valid] * 0.1)

        image_weight = np.ones_like(pc_x, dtype=np.float32)
        if use_image:
            dx = np.zeros_like(pc_x, dtype=np.float32)
            dy = np.zeros_like(pc_y, dtype=np.float32)
            dx[valid] = (uv0[valid] - cam["center_x"]) / cam["center_x"]
            dy[valid] = (uv1[valid] - cam["center_y"]) / cam["center_y"]
            edge_dist = np.zeros_like(pc_x, dtype=np.float32)
            edge_dist[valid] = np.sqrt(dx[valid] * dx[valid] + dy[valid] * dy[valid])
            image_weight[valid] = np.exp(-edge_dist[valid] * edge_dist[valid] * 2.0)

        direction_weight = np.ones_like(pc_x, dtype=np.float32)
        if use_direction:
            if idx == 0:
                direction_weight = np.where(
                    px_grid > 0.0,
                    1.0,
                    np.exp(-np.abs(px_grid) * 0.5),
                )
            elif idx == 1:
                direction_weight = np.where(
                    px_grid < 0.0,
                    1.0,
                    np.exp(-np.abs(px_grid) * 0.5),
                )
            elif idx == 2:
                direction_weight = np.where(
                    py_grid > 0.0,
                    1.0,
                    np.exp(-np.abs(py_grid) * 0.5),
                )
            elif idx == 3:
                direction_weight = np.where(
                    py_grid < 0.0,
                    1.0,
                    np.exp(-np.abs(py_grid) * 0.5),
                )

        weight = dist_weight * image_weight * direction_weight
        weight[~valid] = 0.0
        weights.append(weight.astype(np.float32))

    weights_stack = np.stack(weights, axis=0)
    order = np.argsort(weights_stack, axis=0)
    first_idx = order[-1, :, :]
    second_idx = order[-2, :, :]

    first_val = np.take_along_axis(
        weights_stack, order[-1:, :, :], axis=0
    )[0]
    second_val = np.take_along_axis(
        weights_stack, order[-2:, :, :], axis=0
    )[0]

    total_pixels = ipm_h * ipm_w
    weights_filtered = np.zeros((4, total_pixels), dtype=np.float32)
    flat_first_idx = first_idx.reshape(-1)
    flat_second_idx = second_idx.reshape(-1)
    flat_first_val = first_val.reshape(-1)
    flat_second_val = second_val.reshape(-1)

    pixel_indices = np.arange(total_pixels, dtype=np.int32)
    np.add.at(weights_filtered, (flat_first_idx, pixel_indices), flat_first_val)
    np.add.at(weights_filtered, (flat_second_idx, pixel_indices), flat_second_val)

    weights_filtered = weights_filtered.reshape((4, ipm_h, ipm_w))
    weight_sum = weights_filtered.sum(axis=0)
    mask = weight_sum > 1e-6

    normalized_weights = np.zeros_like(weights_filtered)
    for idx in range(4):
        norm = np.zeros_like(weight_sum)
        norm[mask] = weights_filtered[idx][mask] / weight_sum[mask]
        normalized_weights[idx] = norm

    dominant = np.full((ipm_h, ipm_w), 255, dtype=np.uint8)
    dominant[mask] = first_idx[mask].astype(np.uint8)

    return normalized_weights, dominant


def save_outputs(weights_norm, dominant, output_dir: Path, tag: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]

    for idx in range(4):
        weight = weights_norm[idx]
        png16 = (weight * 65535.0).astype(np.uint16)
        max_val = float(weight.max())
        if max_val > 1e-6:
            color_norm = weight / max_val
        else:
            color_norm = weight
        color_map = (np.clip(color_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"weight_cam{idx}_{tag}.png"), png16)
        colored = cv2.applyColorMap(color_map, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(
            str(output_dir / f"weight_cam{idx}_{tag}_color.png"), colored
        )

    lut = np.zeros((256, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        lut[idx] = color
    lut[255] = (0, 0, 0)
    dominant_color = lut[dominant]
    cv2.imwrite(str(output_dir / f"dominant_{tag}.png"), dominant_color)

    colors = np.array(
        [
            [255.0, 0.0, 0.0],
            [0.0, 255.0, 0.0],
            [0.0, 0.0, 255.0],
            [255.0, 255.0, 0.0],
        ],
        dtype=np.float32,
    )
    fused = np.einsum("cij,ck->ijk", weights_norm, colors)
    fused = np.clip(fused, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(output_dir / f"ipm_debug_{tag}.png"), fused)


def main():
    parser = argparse.ArgumentParser(
        description="Generate IPM seam weight debug maps."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("image_data"),
        help="Directory containing camera calibration YAML files."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory to store generated artifacts."
    )
    parser.add_argument(
        "--tag", type=str, default="synthetic01",
        help="Identifier appended to output file names."
    )
    parser.add_argument(
        "--mode", type=str, default="mixed", choices=tuple(MODE_CONFIGS.keys()),
        help="Weighting mode to emulate (mixed=现有混合策略，baseline=仅距离权重等)."
    )
    args = parser.parse_args()

    cameras = [load_camera(i, args.data_dir) for i in range(4)]
    weights_norm, dominant = compute_weight_maps(
        cameras, mode=args.mode
    )
    save_outputs(weights_norm, dominant, args.output_dir, args.tag)
    print(
        f"Weight maps exported to {args.output_dir} with tag {args.tag} "
        f"(mode={args.mode})"
    )


if __name__ == "__main__":
    main()
