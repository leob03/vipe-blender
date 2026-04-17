#!/usr/bin/env python3
"""
Unproject ViPE metric depth maps to dense per-frame world-space point clouds.
Runs inside the vipe conda environment (needs OpenEXR, numpy, cv2).

Usage:
    python export_depth_pcd.py <output_dir> <video_stem> [--stride N] [--max-frames M]

Output:
    <output_dir>/vipe/<stem>_depth_pcd.npz  with keys:
        coords     (N_total, 3) float32  – world XYZ in ViPE/OpenCV convention
        colors     (N_total, 3) float32  – RGB 0-1
        starts     (N_frames,) int64
        counts     (N_frames,) int64
        frame_inds (N_frames,) int64     – original video frame indices (from pose inds)
"""

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import Imath


def _read_depth_zip(zip_path: Path):
    """Yield (seq_idx, H×W float32 depth) for each EXR in the zip, in order."""
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            seq_idx = int(Path(name).stem)
            with z.open(name) as f:
                try:
                    exr = OpenEXR.InputFile(f)
                    header = exr.header()
                    dw = header["dataWindow"]
                    W = dw.max.x - dw.min.x + 1
                    H = dw.max.y - dw.min.y + 1
                    raw = exr.channels(["Z"])[0]
                    depth = np.frombuffer(raw, dtype=np.float16).reshape(H, W).astype(np.float32)
                except Exception as e:
                    print(f"  [warn] Failed to read frame {seq_idx}: {e}", file=sys.stderr)
                    depth = None
            yield seq_idx, depth


def _unproject(depth: np.ndarray, pose: np.ndarray,
               fx: float, fy: float, cx: float, cy: float,
               stride: int):
    """
    Unproject strided depth pixels to world-space points in ViPE/OpenCV convention.
    Returns (coords (M,3) float32, valid_mask (M,) bool).
    """
    d = depth[::stride, ::stride]
    H_s, W_s = d.shape

    us = np.arange(W_s, dtype=np.float32) * stride
    vs = np.arange(H_s, dtype=np.float32) * stride
    uu, vv = np.meshgrid(us, vs)

    valid = (d > 0) & np.isfinite(d)
    flat_valid = valid.reshape(-1)

    x_cam = ((uu - cx) / fx * d).reshape(-1)[flat_valid]
    y_cam = ((vv - cy) / fy * d).reshape(-1)[flat_valid]
    z_cam = d.reshape(-1)[flat_valid]

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (M, 3)

    R = pose[:3, :3]
    t = pose[:3, 3]
    pts_world = (pts_cam @ R.T + t).astype(np.float32)

    return pts_world, flat_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("stem")
    parser.add_argument("--stride", type=int, default=4,
                        help="Spatial stride for depth sampling (default 4)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Process at most this many frames (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    stem    = args.stem
    stride  = args.stride

    depth_zip = out_dir / "depth"      / f"{stem}.zip"
    pose_npz  = out_dir / "pose"       / f"{stem}.npz"
    intr_npz  = out_dir / "intrinsics" / f"{stem}.npz"
    rgb_mp4   = out_dir / "rgb"        / f"{stem}.mp4"
    out_npz   = out_dir / "vipe"       / f"{stem}_depth_pcd.npz"

    for p in (depth_zip, pose_npz, intr_npz):
        if not p.exists():
            print(f"[export_depth_pcd] Missing: {p}", file=sys.stderr)
            sys.exit(1)

    pose_data = np.load(pose_npz)
    poses     = pose_data["data"]   # (N, 4, 4)
    inds      = pose_data["inds"]   # (N,) original video frame indices

    intr_data = np.load(intr_npz)
    intrs     = intr_data["data"]   # (N, 4) [fx, fy, cx, cy]

    # Open RGB video if available
    cap = None
    has_rgb = rgb_mp4.exists()
    if has_rgb:
        cap = cv2.VideoCapture(str(rgb_mp4))
        if not cap.isOpened():
            has_rgb = False
            cap = None

    all_coords = []
    all_colors = []
    starts_list     = []
    counts_list     = []
    frame_inds_list = []
    total = 0

    depth_iter = list(_read_depth_zip(depth_zip))
    if args.max_frames > 0:
        depth_iter = depth_iter[:args.max_frames]

    n_frames = len(depth_iter)
    print(f"[export_depth_pcd] {n_frames} frames, stride={stride}")

    for seq_idx, depth in depth_iter:
        if seq_idx >= len(poses):
            continue
        if depth is None:
            continue

        pose = poses[seq_idx]
        fx, fy, cx, cy = intrs[seq_idx]
        orig_frame_idx = int(inds[seq_idx])

        coords, flat_valid = _unproject(depth, pose, fx, fy, cx, cy, stride)
        n = len(coords)
        if n == 0:
            continue

        # Sample matching RGB pixels
        if has_rgb:
            cap.set(cv2.CAP_PROP_POS_FRAMES, seq_idx)
            ret, bgr = cap.read()
            if ret:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                rgb_s = rgb[::stride, ::stride].reshape(-1, 3)
                colors = rgb_s[flat_valid]
            else:
                colors = np.ones((n, 3), dtype=np.float32)
        else:
            colors = np.ones((n, 3), dtype=np.float32)

        starts_list.append(total)
        counts_list.append(n)
        frame_inds_list.append(orig_frame_idx)
        all_coords.append(coords)
        all_colors.append(colors.astype(np.float32))
        total += n

        if seq_idx % 10 == 0 or seq_idx == n_frames - 1:
            print(f"  frame {seq_idx:4d} (orig {orig_frame_idx:4d}): {n:6,} pts  total={total:,}")

    if cap is not None:
        cap.release()

    if total == 0:
        print("[export_depth_pcd] No valid points found.", file=sys.stderr)
        sys.exit(1)

    np.savez(
        out_npz,
        coords=np.vstack(all_coords),
        colors=np.vstack(all_colors),
        starts=np.array(starts_list,     dtype=np.int64),
        counts=np.array(counts_list,     dtype=np.int64),
        frame_inds=np.array(frame_inds_list, dtype=np.int64),
    )
    print(f"[export_depth_pcd] Saved {total:,} pts across {len(starts_list)} frames → {out_npz}")


if __name__ == "__main__":
    main()
