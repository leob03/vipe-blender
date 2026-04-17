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


def import_vipe_pointcloud(pcd_path: str, colored: bool = True,
                           point_radius: float = 0.005,
                           mode: str = 'COMBINED',
                           blender_start_frame: int = 1) -> None:
    """
    Import a point cloud produced by export_ply.py or export_depth_pcd.py.

    mode='COMBINED'        — single object, all SLAM points (pcd_path = PLY)
    mode='PER_FRAME'       — one object per SLAM keyframe, animated (pcd_path = PLY)
    mode='PER_FRAME_DENSE' — one object per frame from depth maps (pcd_path = NPZ)
    """
    if mode == 'PER_FRAME_DENSE':
        _import_per_frame_from_packed_npz(
            pcd_path, colored=colored,
            point_radius=point_radius,
            blender_start_frame=blender_start_frame,
            coll_name="VIPE_PointCloud_PerFrame",
            ng_name="VIPE_PerFrameViz",
        )
        return

    if mode == 'PER_FRAME':
        frames_npz = str(Path(pcd_path).with_name(
            Path(pcd_path).stem + "_frames.npz"))
        import_vipe_pointcloud_per_frame(
            pcd_path, frames_npz,
            colored=colored,
            point_radius=point_radius,
            blender_start_frame=blender_start_frame,
        )
        return

    import bpy
    import numpy as np

    obj_name = "VIPE_PointCloud"
    if obj_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

    coords, colors = _parse_ply(ply_path)
    n_verts = len(coords)

    mesh = bpy.data.meshes.new(obj_name)
    mesh.vertices.add(n_verts)
    mesh.vertices.foreach_set("co", coords.flatten())
    mesh.update()

    if colored:
        attr = mesh.attributes.new(name="point_color", type='FLOAT_COLOR', domain='POINT')
        rgba = np.concatenate([colors, np.ones((n_verts, 1), dtype=np.float32)], axis=1)
        attr.data.foreach_set("color", rgba.flatten())

    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    mat = _make_point_material(colored)
    mesh.materials.append(mat)
    _add_point_geo_nodes(obj, mat, point_radius)

    empty_name = "VIPE_GlobalFix"
    if empty_name in bpy.data.objects:
        obj.parent = bpy.data.objects[empty_name]

    print(f"[ViPE] Imported point cloud from {ply_path} ({n_verts} points, radius={point_radius})")


def _import_per_frame_from_packed_npz(npz_path: str,
                                      colored: bool = True,
                                      point_radius: float = 0.005,
                                      blender_start_frame: int = 1,
                                      coll_name: str = "VIPE_PointCloud_PerFrame",
                                      ng_name: str = "VIPE_PerFrameViz") -> None:
    """
    Import per-frame point clouds from a packed NPZ (coords, colors, starts,
    counts, frame_inds). Used for both PER_FRAME_DENSE and as a shared backend.
    """
    import bpy
    import numpy as np

    if not Path(npz_path).exists():
        raise FileNotFoundError(f"Dense PCD file not found: {npz_path}")

    data       = np.load(npz_path)
    coords     = data['coords']      # (N_total, 3)
    colors     = data['colors']      # (N_total, 3)
    starts     = data['starts']
    counts     = data['counts']
    frame_inds = data['frame_inds']
    n_kf       = len(frame_inds)

    _setup_per_frame_collection_and_geonodes(
        coords, colors, starts, counts, frame_inds, n_kf,
        colored, point_radius, blender_start_frame,
        coll_name, ng_name,
    )
    print(f"[ViPE] Imported {n_kf} per-frame point clouds from {Path(npz_path).name}")


def import_vipe_pointcloud_per_frame(ply_path: str, frames_npz_path: str,
                                     colored: bool = True,
                                     point_radius: float = 0.005,
                                     blender_start_frame: int = 1) -> None:
    """
    Create one mesh per SLAM keyframe with keyframed hide_viewport/hide_render.
    """
    import numpy as np

    if not Path(frames_npz_path).exists():
        raise FileNotFoundError(
            f"Per-frame mapping not found: {frames_npz_path}\n"
            "Re-run ViPE to regenerate the _frames.npz file.")

    coords, colors = _parse_ply(ply_path)
    fd         = np.load(frames_npz_path)
    starts     = fd['starts']
    counts     = fd['counts']
    frame_inds = fd['frame_inds']
    n_kf       = len(frame_inds)

    _setup_per_frame_collection_and_geonodes(
        coords, colors, starts, counts, frame_inds, n_kf,
        colored, point_radius, blender_start_frame,
        "VIPE_PointCloud_PerFrame", "VIPE_PerFrameViz",
    )
    print(f"[ViPE] Imported {n_kf} per-frame point clouds (radius={point_radius})")


def _setup_per_frame_collection_and_geonodes(
        coords, colors, starts, counts, frame_inds, n_kf,
        colored, point_radius, blender_start_frame,
        coll_name, ng_name):
    """Shared backend: creates per-frame mesh objects with animated visibility."""
    import bpy
    import numpy as np

    # Reset collection
    if coll_name in bpy.data.collections:
        old = bpy.data.collections[coll_name]
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    coll = bpy.data.collections.new(coll_name)
    bpy.context.scene.collection.children.link(coll)

    global_fix = bpy.data.objects.get("VIPE_GlobalFix")

    # One shared material — ShaderNodeAttribute reads each object's own
    # "point_color" FLOAT_COLOR attribute, so sharing is correct.
    shared_mat = _make_point_material(colored)

    # Shared GeoNodes group: MeshToPoints → SetMaterial → output
    if ng_name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[ng_name])
    ng = bpy.data.node_groups.new(ng_name, type='GeometryNodeTree')
    ng.interface.new_socket(name="Geometry", in_out="INPUT",  socket_type="NodeSocketGeometry")
    rs = ng.interface.new_socket(name="Radius", in_out="INPUT", socket_type="NodeSocketFloat")
    rs.default_value = point_radius
    rs.min_value     = 0.0001
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n_in    = ng.nodes.new('NodeGroupInput')
    n_out   = ng.nodes.new('NodeGroupOutput')
    m2p     = ng.nodes.new('GeometryNodeMeshToPoints')
    set_mat = ng.nodes.new('GeometryNodeSetMaterial')
    set_mat.inputs[2].default_value = shared_mat
    ng.links.new(n_in.outputs['Geometry'],    m2p.inputs['Mesh'])
    ng.links.new(n_in.outputs['Radius'],      m2p.inputs['Radius'])
    ng.links.new(m2p.outputs['Points'],       set_mat.inputs['Geometry'])
    ng.links.new(set_mat.outputs['Geometry'], n_out.inputs['Geometry'])
    radius_id = next((item.identifier for item in ng.interface.items_tree
                      if item.name == "Radius"), None)

    for i in range(n_kf):
        start = int(starts[i])
        count = int(counts[i])
        fidx  = int(frame_inds[i])

        kf_coords = coords[start:start + count]
        if len(kf_coords) == 0:
            continue
        kf_colors = colors[start:start + count] if colored else None

        obj_name = f"VIPE_PCFrame_{fidx:05d}"
        mesh = bpy.data.meshes.new(obj_name)
        mesh.vertices.add(len(kf_coords))
        mesh.vertices.foreach_set("co", kf_coords.flatten())
        mesh.update()

        if colored and kf_colors is not None:
            attr = mesh.attributes.new(name="point_color", type='FLOAT_COLOR', domain='POINT')
            rgba = np.concatenate([kf_colors,
                                   np.ones((len(kf_colors), 1), dtype=np.float32)], axis=1)
            attr.data.foreach_set("color", rgba.flatten())

        obj = bpy.data.objects.new(obj_name, mesh)
        coll.objects.link(obj)

        geo_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
        geo_mod.node_group = ng
        if radius_id:
            geo_mod[radius_id] = float(point_radius)

        if global_fix:
            obj.parent = global_fix

        bf = blender_start_frame + fidx
        next_bf = (blender_start_frame + int(frame_inds[i + 1])) if i + 1 < n_kf else bf + 1
        _keyframe_visibility(obj, bf, next_bf)


def _keyframe_visibility(obj, show_frame: int, hide_frame: int) -> None:
    """Animate obj to be visible only on [show_frame, hide_frame)."""
    import bpy

    obj.hide_viewport = True
    obj.hide_render   = True
    obj.keyframe_insert(data_path="hide_viewport", frame=1)
    obj.keyframe_insert(data_path="hide_render",   frame=1)

    obj.hide_viewport = False
    obj.hide_render   = False
    obj.keyframe_insert(data_path="hide_viewport", frame=show_frame)
    obj.keyframe_insert(data_path="hide_render",   frame=show_frame)

    obj.hide_viewport = True
    obj.hide_render   = True
    obj.keyframe_insert(data_path="hide_viewport", frame=hide_frame)
    obj.keyframe_insert(data_path="hide_render",   frame=hide_frame)

    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            if fc.data_path in ("hide_viewport", "hide_render"):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'


def _parse_ply(ply_path: str):
    """Parse a binary PLY file, returning (coords, colors) as float32 arrays."""
    import numpy as np

    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        verts = plydata["vertex"].data
        coords = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
        if {"red", "green", "blue"}.issubset(verts.dtype.names):
            colors = np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.float32) / 255.0
        else:
            colors = np.ones((len(coords), 3), dtype=np.float32)
        return coords, colors
    except ImportError:
        pass

    # Manual fallback parser (no plyfile dependency)
    with open(ply_path, 'rb') as f:
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b"end_header":
                break

        header = header_bytes.decode('utf-8')
        n_verts = int(next(l for l in header.splitlines()
                           if l.startswith("element vertex")).split()[-1])

        type_map = {
            'double': np.float64, 'float': np.float32,
            'uchar': np.uint8,    'uint8': np.uint8,
            'int': np.int32,      'uint': np.uint32,
        }
        props = []
        for l in header.splitlines():
            if l.startswith("property"):
                parts = l.split()
                props.append((parts[2], type_map.get(parts[1], np.float32)))

        dtype = np.dtype(props)
        data  = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype)

    coords = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
    if all(c in data.dtype.names for c in ('red', 'green', 'blue')):
        colors = np.stack([data['red'], data['green'], data['blue']], axis=1).astype(np.float32) / 255.0
    else:
        colors = np.ones((len(coords), 3), dtype=np.float32)

    return coords, colors


def _make_point_material(colored: bool):
    import bpy

    mat = bpy.data.materials.new(name="VIPE_PointCloud_Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    emit = nt.nodes.new("ShaderNodeEmission")
    emit.location = (0, 0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (250, 0)

    if colored:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "point_color"
        attr.location = (-300, 0)
        nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    else:
        emit.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _add_point_geo_nodes(obj, mat, point_radius: float = 0.005):
    import bpy

    geo_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
    ng = bpy.data.node_groups.new(name="VIPE_PointCloudViz", type='GeometryNodeTree')
    geo_mod.node_group = ng

    ng.interface.new_socket(name="Geometry", in_out="INPUT",  socket_type="NodeSocketGeometry")
    radius_sock = ng.interface.new_socket(name="Radius", in_out="INPUT", socket_type="NodeSocketFloat")
    radius_sock.default_value = point_radius
    radius_sock.min_value     = 0.0001
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    n_in  = ng.nodes.new('NodeGroupInput')
    n_out = ng.nodes.new('NodeGroupOutput')
    m2p   = ng.nodes.new('GeometryNodeMeshToPoints')
    set_mat = ng.nodes.new('GeometryNodeSetMaterial')
    set_mat.inputs[2].default_value = mat

    ng.links.new(n_in.outputs['Geometry'],    m2p.inputs['Mesh'])
    ng.links.new(n_in.outputs['Radius'],      m2p.inputs['Radius'])
    ng.links.new(m2p.outputs['Points'],       set_mat.inputs['Geometry'])
    ng.links.new(set_mat.outputs['Geometry'], n_out.inputs['Geometry'])

    for item in ng.interface.items_tree:
        if item.name == "Radius":
            geo_mod[item.identifier] = float(point_radius)
            break
