"""
Import ViPE results into Blender.

Coordinate-system conventions
------------------------------
ViPE (inherited from DROID-SLAM / OpenCV):
  Camera local frame  — X right, Y down,  Z forward
  World frame         — Y down,  Z forward (same as camera at rest)

Blender:
  Camera local frame  — X right, Y up,    Z back
  World frame         — X right, Y forward, Z up

Two-step correction (identical to the DROID-SLAM blender addon):
  1. Per-pose: right-multiply rotation by M = diag(1, -1, -1) to flip the
     camera's Y and Z axes (down→up, forward→back).
  2. Global:  parent to an empty with –90° rotation around X, which
     rotates the whole world from (Z forward, Y down) to (Z up, Y forward).
"""

import math
import glob
from pathlib import Path

from mathutils import Matrix, Vector


# OpenCV camera axes → Blender camera axes
_M = Matrix(((1,  0,  0),
             (0, -1,  0),
             (0,  0, -1)))


def import_vipe_camera(output_dir: str, blender_start_frame: int = 1,
                       stem: str = None) -> int:
    """
    Load pose/*.npz and intrinsics/*.npz from *output_dir* and create an
    animated Blender camera named "VIPE_Camera".

    If *stem* is given, looks for files matching that stem exactly.
    Returns the number of poses imported.
    """
    import bpy
    import numpy as np

    out = Path(output_dir)

    if stem:
        pose_candidates = [str(out / "pose" / f"{stem}.npz")]
        intr_candidates = [str(out / "intrinsics" / f"{stem}.npz")]
    else:
        pose_candidates = sorted(glob.glob(str(out / "pose" / "*.npz")))
        intr_candidates = sorted(glob.glob(str(out / "intrinsics" / "*.npz")))

    pose_files = [p for p in pose_candidates if Path(p).exists()]
    intr_files = [p for p in intr_candidates if Path(p).exists()]

    if not pose_files:
        raise FileNotFoundError(f"No pose .npz found in {out / 'pose'}"
                                + (f" for stem '{stem}'" if stem else ""))

    pose_data = np.load(pose_files[0])
    poses = pose_data["data"]   # (N, 4, 4)  camera-to-world
    inds  = pose_data["inds"]   # (N,)       original frame indices

    fx = fy = cx = cy = 0.0
    if intr_files:
        intr_data = np.load(intr_files[0])
        intrs = intr_data["data"]   # (N, 4)  [fx, fy, cx, cy]
        fx, fy, cx, cy = float(intrs[0, 0]), float(intrs[0, 1]), \
                         float(intrs[0, 2]), float(intrs[0, 3])

    cam_name = "VIPE_Camera"
    if cam_name in bpy.data.objects:
        cam_obj  = bpy.data.objects[cam_name]
        cam_data = cam_obj.data
    else:
        cam_data = bpy.data.cameras.new(name=cam_name)
        cam_obj  = bpy.data.objects.new(cam_name, cam_data)
        bpy.context.collection.objects.link(cam_obj)

    if fx > 0:
        W = round(cx * 2)
        H = round(cy * 2)
        cam_data.type          = 'PERSP'
        cam_data.sensor_fit    = 'HORIZONTAL'
        cam_data.sensor_width  = float(W)
        cam_data.sensor_height = float(H)
        cam_data.lens          = float(fx)
        bpy.context.scene.render.resolution_x = W
        bpy.context.scene.render.resolution_y = H

    bpy.context.scene.camera = cam_obj
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.animation_data_clear()

    for i, frame_idx in enumerate(inds):
        T = poses[i]
        R_c2w = Matrix([T[r, :3].tolist() for r in range(3)])
        t_c2w = Vector(T[:3, 3].tolist())

        mat = (R_c2w @ _M).to_4x4()
        mat.translation = t_c2w

        cam_obj.matrix_world = mat

        blender_frame = blender_start_frame + int(frame_idx)
        cam_obj.keyframe_insert(data_path="location",            frame=blender_frame)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)

    empty_name = "VIPE_GlobalFix"
    if empty_name in bpy.data.objects:
        empty = bpy.data.objects[empty_name]
    else:
        empty = bpy.data.objects.new(empty_name, None)
        bpy.context.collection.objects.link(empty)

    empty.rotation_mode  = 'XYZ'
    empty.rotation_euler = (math.radians(-90.0), 0.0, 0.0)
    cam_obj.parent = empty

    bpy.context.scene.frame_start = blender_start_frame
    bpy.context.scene.frame_end   = blender_start_frame + int(inds[-1])

    n = len(inds)
    print(f"[ViPE] Imported {n} poses from {pose_files[0]}")
    return n


def import_vipe_pointcloud(ply_path: str, colored: bool = True) -> None:
    """
    Import a PLY point cloud produced by export_ply.py and parent it to
    the VIPE_GlobalFix empty so it lives in the same coordinate space as
    the camera.
    """
    import bpy

    try:
        bpy.ops.wm.ply_import(filepath=ply_path)
    except AttributeError:
        bpy.ops.import_mesh.ply(filepath=ply_path)

    obj = bpy.context.active_object

    empty_name = "VIPE_GlobalFix"
    if empty_name in bpy.data.objects and obj:
        obj.parent = bpy.data.objects[empty_name]

    if colored and obj:
        _apply_vertex_color_material(obj)

    print(f"[ViPE] Imported point cloud from {ply_path} (colored={colored})")


def _apply_vertex_color_material(obj) -> None:
    import bpy

    mesh = obj.data
    attr_name = mesh.color_attributes[0].name if mesh.color_attributes else "Col"

    mat = bpy.data.materials.new(name="VIPE_PointCloud")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    col_node  = nodes.new("ShaderNodeVertexColor")
    col_node.layer_name = attr_name

    emit_node = nodes.new("ShaderNodeEmission")
    out_node  = nodes.new("ShaderNodeOutputMaterial")

    links.new(col_node.outputs["Color"],     emit_node.inputs["Color"])
    links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

    obj.data.materials.append(mat)
