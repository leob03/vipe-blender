import os
import select
import subprocess
import shlex
from pathlib import Path

import bpy

from .properties import DEPTH_MODEL_MAP, DEPTH_ALIGN_MAP


# Module-level handles shared between operator callbacks and the timer
_process  = None
_timer    = None
_log_file = None


# ---------------------------------------------------------------------------
# Timer callback — polls the subprocess every second
# ---------------------------------------------------------------------------

def _drain_stdout():
    """Read all currently available bytes from the subprocess stdout."""
    if _process is None or _process.stdout is None:
        return
    while True:
        ready = select.select([_process.stdout], [], [], 0)[0]
        if not ready:
            break
        chunk = os.read(_process.stdout.fileno(), 4096)
        if not chunk:
            break
        text = chunk.decode('utf-8', errors='replace')
        if _log_file:
            _log_file.write(text)
            _log_file.flush()
        print(text, end='', flush=True)


def _poll_process():
    global _process, _timer, _log_file

    if _process is None:
        return None  # cancel timer

    _drain_stdout()

    retcode = _process.poll()
    if retcode is None:
        return 1.0  # still running

    # Drain any remaining output after process exits
    _drain_stdout()

    if _log_file:
        _log_file.close()
        _log_file = None

    props = bpy.context.scene.vipe

    if retcode == 0:
        _resolve_ply_path(props)
        props.status = "Importing…"
        _import_results(props)
        props.status = "Done"
    else:
        props.status = f"Error (exit {retcode}) — see log"

    _process = None
    _timer   = None
    return None  # cancel timer


def _resolve_ply_path(props):
    """Find the PLY and dense depth PCD for the current run using the stored stem."""
    stem = props.last_stem
    out  = props.last_output_dir
    if not stem or not out:
        return
    ply = os.path.join(out, "vipe", f"{stem}_slam_map.ply")
    if os.path.exists(ply):
        props.last_ply_path = ply
    depth_pcd = os.path.join(out, "vipe", f"{stem}_depth_pcd.npz")
    if os.path.exists(depth_pcd):
        props.last_depth_pcd_path = depth_pcd


def _import_results(props):
    """Auto-import camera and (optionally) point cloud after a successful run."""
    out  = props.last_output_dir
    stem = props.last_stem
    if not out or not os.path.isdir(out):
        return

    try:
        from .importer import import_vipe_camera
        import_vipe_camera(out, blender_start_frame=props.blender_start_frame,
                           stem=stem)
    except Exception as e:
        print(f"[ViPE] Camera import failed: {e}")

    if not props.import_pointcloud:
        return

    from .importer import import_vipe_pointcloud
    mode = props.pointcloud_mode

    if mode == 'PER_FRAME_DENSE':
        pcd_path = props.last_depth_pcd_path
        if pcd_path and os.path.exists(pcd_path):
            try:
                import_vipe_pointcloud(pcd_path, colored=props.colored_pointcloud,
                                       point_radius=props.point_radius,
                                       mode=mode,
                                       blender_start_frame=props.blender_start_frame)
            except Exception as e:
                print(f"[ViPE] Dense point cloud import failed: {e}")
        else:
            print("[ViPE] Dense PCD not found — re-run to generate it.")
    elif props.save_slam_map:
        ply_path = props.last_ply_path
        if ply_path and os.path.exists(ply_path):
            try:
                import_vipe_pointcloud(ply_path, colored=props.colored_pointcloud,
                                       point_radius=props.point_radius,
                                       mode=mode,
                                       blender_start_frame=props.blender_start_frame)
            except Exception as e:
                print(f"[ViPE] Point cloud import failed: {e}")


# ---------------------------------------------------------------------------
# Helper: build the Hydra override list for run.py
# ---------------------------------------------------------------------------

def _build_hydra_args(props, input_path: str, output_dir: str) -> list:
    """Return a list of Hydra override strings to pass to run.py."""
    is_image_dir = props.input_type == 'IMAGE_DIR'

    args = []

    # Stream config
    if is_image_dir:
        args += ["streams=frame_dir_stream", f"streams.base_path={input_path}"]
    else:
        args += [f"streams.base_path={input_path}"]

    # Frame range / skip
    if props.frame_start != 0:
        args.append(f"streams.frame_start={props.frame_start}")
    args.append(f"streams.frame_end={props.frame_end}")
    if props.frame_skip != 1:
        args.append(f"streams.frame_skip={props.frame_skip}")

    # Pipeline preset
    args.append(f"pipeline={props.pipeline_preset}")

    # Output
    args += [
        f"pipeline.output.path={output_dir}",
        "pipeline.output.save_artifacts=true",
        f"pipeline.output.save_slam_map={'true' if props.save_slam_map else 'false'}",
        f"pipeline.output.save_viz={'true' if props.save_viz else 'false'}",
    ]

    # Camera type
    args.append(f"pipeline.init.camera_type={props.camera_type}")

    # Intrinsics — calib file mode reads fx/fy/cx/cy here and passes as manual
    mode = props.intrinsics_mode
    if mode == 'calib':
        fx, fy, cx, cy = _read_calib_file(bpy.path.abspath(props.calib_file))
        args.append("pipeline.init.intrinsics=manual")
        args.append(f"+pipeline.init.fx={fx:.4f}")
        args.append(f"+pipeline.init.fy={fy:.4f}")
        args.append(f"+pipeline.init.cx={cx:.4f}")
        args.append(f"+pipeline.init.cy={cy:.4f}")
        args.append("pipeline.slam.optimize_intrinsics=false")
    else:
        args.append(f"pipeline.init.intrinsics={mode}")
        if mode == 'fov':
            args.append(f"+pipeline.init.fov_y_deg={props.fov_y_deg:.4f}")
            args.append("pipeline.slam.optimize_intrinsics=false")
        elif mode == 'manual':
            args.append(f"+pipeline.init.fx={props.fx:.4f}")
            args.append(f"+pipeline.init.fy={props.fy:.4f}")
            args.append(f"+pipeline.init.cx={props.cx:.4f}")
            args.append(f"+pipeline.init.cy={props.cy:.4f}")
            args.append("pipeline.slam.optimize_intrinsics=false")

    depth_model = DEPTH_MODEL_MAP[props.depth_model]
    args.append(f"pipeline.slam.keyframe_depth={depth_model}")

    depth_align = DEPTH_ALIGN_MAP[props.depth_align_model]
    args.append(f"pipeline.post.depth_align_model={depth_align}")

    return args


def _read_calib_file(path: str):
    """Parse 'fx fy cx cy' from the first line of a DROID-SLAM style calib file."""
    import numpy as np
    vals = np.loadtxt(path)
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


# ---------------------------------------------------------------------------
# VIPE_OT_Run
# ---------------------------------------------------------------------------

class VIPE_OT_Run(bpy.types.Operator):
    bl_idname      = "vipe.run"
    bl_label       = "Run ViPE"
    bl_description = "Launch ViPE in the background and import results when done"

    def execute(self, context):
        global _process, _timer, _log_file

        if _process is not None and _process.poll() is None:
            self.report({'WARNING'}, "ViPE is already running.")
            return {'CANCELLED'}

        props = context.scene.vipe

        # --- validate ---
        if not props.vipe_dir:
            self.report({'ERROR'}, "ViPE Directory is not set.")
            return {'CANCELLED'}
        if not props.input_path:
            self.report({'ERROR'}, "No input path specified.")
            return {'CANCELLED'}
        if props.intrinsics_mode == 'calib' and not props.calib_file:
            self.report({'ERROR'}, "Calibration file not set.")
            return {'CANCELLED'}

        vipe_dir   = bpy.path.abspath(props.vipe_dir)
        input_path = bpy.path.abspath(props.input_path)
        output_dir = bpy.path.abspath(props.output_dir) if props.output_dir else \
                     os.path.join(vipe_dir, "vipe_results")

        os.makedirs(output_dir, exist_ok=True)

        stem = Path(input_path).stem
        props.last_output_dir = output_dir
        props.last_stem           = stem
        props.last_log_path       = os.path.join(output_dir, f"{stem}_vipe.log")
        props.last_ply_path       = ""
        props.last_depth_pcd_path = ""

        # --- locate conda ---
        try:
            conda_base = subprocess.check_output(
                ["conda", "info", "--base"], text=True
            ).strip()
        except Exception as e:
            self.report({'ERROR'}, f"Could not locate conda: {e}")
            return {'CANCELLED'}

        conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")

        # --- build command ---
        hydra_args = _build_hydra_args(props, input_path, output_dir)
        hydra_str  = " ".join(shlex.quote(a) for a in hydra_args)

        run_py          = os.path.join(vipe_dir, "run.py")
        addon_dir       = os.path.dirname(os.path.abspath(__file__))
        export_ply_py   = os.path.join(addon_dir, "export_ply.py")
        export_depth_py = os.path.join(addon_dir, "export_depth_pcd.py")

        vipe_cmd = f"python {shlex.quote(run_py)} {hydra_str}"

        if props.import_pointcloud:
            mode = props.pointcloud_mode
            if props.save_slam_map and mode in ('COMBINED', 'PER_FRAME'):
                vipe_cmd += (f" && python {shlex.quote(export_ply_py)}"
                             f" {shlex.quote(output_dir)} {shlex.quote(stem)}")
            if mode == 'PER_FRAME_DENSE':
                vipe_cmd += (f" && python {shlex.quote(export_depth_py)}"
                             f" {shlex.quote(output_dir)} {shlex.quote(stem)}"
                             f" --stride {props.depth_pcd_stride}")

        bash_cmd = (
            f"source {shlex.quote(conda_sh)} && "
            f"conda activate {shlex.quote(props.conda_env)} && "
            f"TORCH_LIB=$(python -c "
            f"\"import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))\") && "
            f"NVIDIA_LIBS=$(python -c \"import site, os, glob; "
            f"dirs=[d for sp in site.getsitepackages() for d in glob.glob(os.path.join(sp,'nvidia','*','lib'))]; "
            f"print(':'.join(dirs))\") && "
            f"export LD_LIBRARY_PATH=\"$TORCH_LIB:$NVIDIA_LIBS\" && "
            f"cd {shlex.quote(vipe_dir)} && "
            f"{vipe_cmd}"
        )

        env = os.environ.copy()
        for var in ('LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
                    'CUDA_HOME', 'CUDA_ROOT', 'CUDA_PATH'):
            env.pop(var, None)

        log_path  = props.last_log_path
        _log_file = open(log_path, 'w')
        _log_file.write(f"Command: bash -c '{bash_cmd}'\n\n")
        _log_file.flush()

        _process = subprocess.Popen(
            ["bash", "-c", bash_cmd],
            cwd=vipe_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
        )

        props.status = "Running…"
        _timer = bpy.app.timers.register(_poll_process, first_interval=2.0)

        self.report({'INFO'}, f"ViPE started. Log: {log_path}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# VIPE_OT_Cancel
# ---------------------------------------------------------------------------

class VIPE_OT_Cancel(bpy.types.Operator):
    bl_idname      = "vipe.cancel"
    bl_label       = "Cancel"
    bl_description = "Kill the running ViPE process"

    def execute(self, context):
        global _process, _log_file
        if _process is not None and _process.poll() is None:
            _process.terminate()
            context.scene.vipe.status = "Cancelled"
        if _log_file:
            _log_file.close()
            _log_file = None
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# VIPE_OT_OpenLog
# ---------------------------------------------------------------------------

class VIPE_OT_OpenLog(bpy.types.Operator):
    bl_idname      = "vipe.open_log"
    bl_label       = "Open Log"
    bl_description = "Load the ViPE log into Blender's text editor"

    def execute(self, context):
        log_path = context.scene.vipe.last_log_path
        if not log_path or not os.path.exists(log_path):
            self.report({'WARNING'}, "No log file found.")
            return {'CANCELLED'}

        text_name = os.path.basename(log_path)
        if text_name in bpy.data.texts:
            bpy.data.texts.remove(bpy.data.texts[text_name])
        with open(log_path) as f:
            content = f.read()
        text = bpy.data.texts.new(text_name)
        text.write(content)

        for area in context.screen.areas:
            if area.type == 'TEXT_EDITOR':
                area.spaces.active.text = text
                break

        self.report({'INFO'}, f"Log loaded: {text_name}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# VIPE_OT_ImportCamera  (manual re-import)
# ---------------------------------------------------------------------------

class VIPE_OT_ImportCamera(bpy.types.Operator):
    bl_idname      = "vipe.import_camera"
    bl_label       = "Import Camera"
    bl_description = "Import the ViPE camera trajectory from the output directory"

    directory: bpy.props.StringProperty(subtype='DIR_PATH')

    def invoke(self, context, event):
        props = context.scene.vipe
        if props.last_output_dir:
            self.directory = props.last_output_dir
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from .importer import import_vipe_camera
        props = context.scene.vipe
        try:
            n = import_vipe_camera(self.directory,
                                   blender_start_frame=props.blender_start_frame,
                                   stem=props.last_stem or None)
            self.report({'INFO'}, f"Imported {n} camera poses.")
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# VIPE_OT_ImportPointCloud  (manual re-import)
# ---------------------------------------------------------------------------

class VIPE_OT_ImportPointCloud(bpy.types.Operator):
    bl_idname      = "vipe.import_pointcloud"
    bl_label       = "Import Point Cloud"
    bl_description = "Import a ViPE PLY point cloud (produced by export_ply.py)"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')

    def invoke(self, context, event):
        props = context.scene.vipe
        if props.last_ply_path:
            self.filepath = props.last_ply_path
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from .importer import import_vipe_pointcloud
        props = context.scene.vipe
        try:
            import_vipe_pointcloud(self.filepath, colored=props.colored_pointcloud,
                                   point_radius=props.point_radius,
                                   mode=props.pointcloud_mode,
                                   blender_start_frame=props.blender_start_frame)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


def _set_radius_on_obj(obj, radius: float) -> int:
    """Set GeoNodes Radius on obj. Returns 1 if updated, 0 otherwise."""
    mod = obj.modifiers.get("GeometryNodes")
    if mod is None or mod.node_group is None:
        return 0
    for item in mod.node_group.interface.items_tree:
        if item.name == "Radius":
            mod[item.identifier] = float(radius)
            obj.data.update()
            return 1
    return 0


# ---------------------------------------------------------------------------
# VIPE_OT_SetPointRadius  (update radius on already-imported point cloud)
# ---------------------------------------------------------------------------

class VIPE_OT_SetPointRadius(bpy.types.Operator):
    bl_idname      = "vipe.set_point_radius"
    bl_label       = "Set Radius"
    bl_description = "Update the display radius of the imported ViPE point cloud"

    def execute(self, context):
        import bpy as _bpy
        radius = context.scene.vipe.point_radius
        updated = 0

        # Combined mode object
        obj = _bpy.data.objects.get("VIPE_PointCloud")
        if obj:
            updated += _set_radius_on_obj(obj, radius)

        # Per-frame collection objects
        coll = _bpy.data.collections.get("VIPE_PointCloud_PerFrame")
        if coll:
            for obj in coll.objects:
                updated += _set_radius_on_obj(obj, radius)

        if updated == 0:
            self.report({'WARNING'}, "No VIPE point cloud objects found in scene.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Radius updated on {updated} object(s).")
        return {'FINISHED'}


CLASSES = [
    VIPE_OT_Run,
    VIPE_OT_Cancel,
    VIPE_OT_OpenLog,
    VIPE_OT_ImportCamera,
    VIPE_OT_ImportPointCloud,
    VIPE_OT_SetPointRadius,
]
