#!/usr/bin/env python3
"""
Standalone helper: converts a ViPE slam_map.pt to a binary PLY point cloud.

Run this script inside the vipe conda environment — it requires torch.
Usage:
    python export_ply.py <output_dir> <video_stem>

The script looks for <output_dir>/vipe/<video_stem>_slam_map.pt and writes
<output_dir>/vipe/<video_stem>_slam_map.ply alongside it.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()

    xyz_f32 = xyz.astype(np.float32)
    rgb_u8  = (rgb * 255).clip(0, 255).astype(np.uint8)

    with open(path, "wb") as f:
        f.write(header)
        for i in range(n):
            f.write(struct.pack("<fff", xyz_f32[i, 0], xyz_f32[i, 1], xyz_f32[i, 2]))
            f.write(struct.pack("<BBB", rgb_u8[i, 0],  rgb_u8[i, 1],  rgb_u8[i, 2]))


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: export_ply.py <output_dir> <video_stem>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    stem       = sys.argv[2]

    slam_map_path = output_dir / "vipe" / f"{stem}_slam_map.pt"
    if not slam_map_path.exists():
        print(f"[export_ply] Not found: {slam_map_path}", file=sys.stderr)
        sys.exit(1)

    ply_path = slam_map_path.with_suffix(".ply")

    print(f"[export_ply] Loading {slam_map_path} ...")
    data = torch.load(slam_map_path, map_location="cpu", weights_only=False)

    xyz = data["dense_disp_xyz"].numpy()
    rgb = data["dense_disp_rgb"].float().numpy()

    print(f"[export_ply] Writing {len(xyz):,} points → {ply_path}")
    write_ply(ply_path, xyz, rgb)

    # Per-keyframe index mapping for animated import
    if "dense_disp_packinfo" in data and "dense_disp_frame_inds" in data:
        packinfo   = data["dense_disp_packinfo"].squeeze(1).numpy()  # (n_kf, 2)
        frame_inds = np.array(data["dense_disp_frame_inds"], dtype=np.int64)
        frames_npz = slam_map_path.with_name(slam_map_path.stem + "_frames.npz")
        np.savez(
            frames_npz,
            starts=packinfo[:, 0].astype(np.int64),
            counts=packinfo[:, 1].astype(np.int64),
            frame_inds=frame_inds,
        )
        print(f"[export_ply] Wrote {len(frame_inds)} keyframe entries → {frames_npz}")

    print(f"[export_ply] Done.")


if __name__ == "__main__":
    main()
