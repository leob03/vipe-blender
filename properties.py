import os
import bpy
from bpy.props import (
    StringProperty, BoolProperty, IntProperty, FloatProperty, EnumProperty,
)


PIPELINE_ITEMS = [
    ('default',     'Default',      'Standard pipeline (UniDepth-L + SVDA alignment)'),
    ('dav3',        'DAv3',         'Uses Depth-Anything-V3 for depth (requires dav3 extra)'),
    ('no_vda',      'No VDA',       'Default pipeline without VideoDepthAnything'),
    ('wide_angle',  'Wide Angle',   'Optimised for fisheye / wide-angle cameras (MEI model)'),
    ('static_vda',  'Static VDA',   'Static VideoDepthAnything variant'),
]

CAMERA_TYPE_ITEMS = [
    ('pinhole', 'Pinhole',          'Standard perspective camera'),
    ('mei',     'Wide Angle (MEI)', 'Fisheye / wide-angle camera using the Mei model'),
]

INTRINSICS_ITEMS = [
    ('geocalib', 'GeoCalib (auto)',      'AI-powered calibration — samples a few frames to estimate FOV'),
    ('fov',      'Manual FOV',           'Specify the vertical field-of-view angle in degrees'),
    ('manual',   'Manual (fx fy cx cy)', 'Provide camera intrinsics directly — skips all auto-calibration'),
    ('calib',    'Calibration File',     'Read fx, fy, cx, cy from a text file (same format as DROID-SLAM)'),
]

DEPTH_MODEL_ITEMS = [
    ('UNIDEPTH_L',     'UniDepth-L',         'Lightweight, general-purpose (default)'),
    ('METRIC3D',       'Metric3D',           'Scale-aware metric depth'),
    ('METRIC3D_SMALL', 'Metric3D Small',     'Faster/smaller variant of Metric3D'),
    ('DAV3',           'Depth-Anything-V3',  'State-of-the-art; requires the dav3 extra'),
]

DEPTH_MODEL_MAP = {
    'UNIDEPTH_L':     'unidepth-l',
    'METRIC3D':       'metric3d',
    'METRIC3D_SMALL': 'metric3d-small',
    'DAV3':           'dav3',
}

DEPTH_ALIGN_ITEMS = [
    ('ADAPTIVE_UNIDEPTH_L_SVDA', 'Adaptive UniDepth-L + SVDA', 'Default — robust scale/shift alignment'),
    ('ADAPTIVE_UNIDEPTH_L',      'Adaptive UniDepth-L',        'Alignment without SVDA'),
    ('MVD_DAV3',                 'MVD + DAv3',                 'Multi-view diffusion; use with DAv3 pipeline'),
    ('NONE',                     'None',                       'Skip depth alignment'),
]

DEPTH_ALIGN_MAP = {
    'ADAPTIVE_UNIDEPTH_L_SVDA': 'adaptive_unidepth-l_svda',
    'ADAPTIVE_UNIDEPTH_L':      'adaptive_unidepth-l',
    'MVD_DAV3':                 'mvd_dav3',
    'NONE':                     'null',
}


class VipeProperties(bpy.types.PropertyGroup):

    # ------------------------------------------------------------------ paths
    vipe_dir: StringProperty(
        name="ViPE Directory",
        description="Root of the ViPE repository (contains run.py)",
        default=os.path.expanduser("~/dev-lbringer/camera_tracking/vipe"),
        subtype='DIR_PATH',
    )
    conda_env: StringProperty(
        name="Conda Env",
        description="Name of the conda environment that has vipe installed",
        default="vipe",
    )
    input_type: EnumProperty(
        name="Input Type",
        items=[
            ('VIDEO',     'Video File',      'Single video file (.mp4, .mov, …)'),
            ('IMAGE_DIR', 'Image Directory', 'Directory of image frames'),
        ],
        default='VIDEO',
    )
    input_path: StringProperty(
        name="Input",
        description="Video file or image directory to process",
        subtype='FILE_PATH',
    )
    output_dir: StringProperty(
        name="Output Dir",
        description="Directory where ViPE saves results",
        subtype='DIR_PATH',
    )

    # --------------------------------------------------- intrinsics (prominent)
    intrinsics_mode: EnumProperty(
        name="Intrinsics",
        items=INTRINSICS_ITEMS,
        default='geocalib',
    )
    fov_y_deg: FloatProperty(
        name="FOV Y (°)",
        description="Vertical field-of-view in degrees",
        default=45.0, min=5.0, max=170.0, step=10, precision=2,
    )
    fx: FloatProperty(name="fx", description="Horizontal focal length in pixels", default=1000.0, min=1.0)
    fy: FloatProperty(name="fy", description="Vertical focal length in pixels",   default=1000.0, min=1.0)
    cx: FloatProperty(name="cx", description="Principal point X in pixels",       default=960.0,  min=0.0)
    cy: FloatProperty(name="cy", description="Principal point Y in pixels",       default=540.0,  min=0.0)
    calib_file: StringProperty(
        name="Calib File",
        description="Text file with 'fx fy cx cy' on the first line (DROID-SLAM format)",
        subtype='FILE_PATH',
    )

    # -------------------------------------------------------- pipeline preset
    pipeline_preset: EnumProperty(
        name="Pipeline",
        description="Named pipeline configuration to use",
        items=PIPELINE_ITEMS,
        default='default',
    )

    # ----------------------------------------------------- granular overrides
    show_overrides: BoolProperty(
        name="Show Overrides",
        description="Expand granular pipeline overrides",
        default=False,
    )
    camera_type: EnumProperty(
        name="Camera Type",
        items=CAMERA_TYPE_ITEMS,
        default='pinhole',
    )
    depth_model: EnumProperty(
        name="Depth Model",
        description="Depth estimator used by the SLAM keyframe backend",
        items=DEPTH_MODEL_ITEMS,
        default='UNIDEPTH_L',
    )
    depth_align_model: EnumProperty(
        name="Depth Alignment",
        description="Post-processing depth alignment model",
        items=DEPTH_ALIGN_ITEMS,
        default='ADAPTIVE_UNIDEPTH_L_SVDA',
    )
    save_slam_map: BoolProperty(
        name="Save SLAM Map",
        description="Save the dense SLAM point cloud (required for point cloud import)",
        default=True,
    )
    save_viz: BoolProperty(
        name="Save Visualisation Videos",
        description="Write RGB + depth visualisation videos alongside the results",
        default=False,
    )

    # ------------------------------------------------------- stream / frames
    show_frame_range: BoolProperty(
        name="Frame Range",
        description="Expand frame range options",
        default=False,
    )
    frame_start: IntProperty(
        name="Frame Start",
        description="First frame to process from the input",
        default=0, min=0,
    )
    frame_end: IntProperty(
        name="Frame End",
        description="Last frame to process (-1 = all frames)",
        default=-1, min=-1,
    )
    frame_skip: IntProperty(
        name="Frame Skip",
        description="Process every N-th frame (1 = every frame)",
        default=1, min=1,
    )

    # -------------------------------------------------------- import options
    blender_start_frame: IntProperty(
        name="Blender Start Frame",
        description="Blender timeline frame that corresponds to input frame 0",
        default=1, min=0,
    )
    import_pointcloud: BoolProperty(
        name="Import Point Cloud",
        description="Import the SLAM map as a coloured point cloud after processing",
        default=True,
    )
    colored_pointcloud: BoolProperty(
        name="Vertex Colours",
        description="Apply an emission material driven by the point cloud vertex colours",
        default=True,
    )

    # --------------------------------------------------------- runtime state
    status: StringProperty(name="Status", default="Ready")
    last_output_dir: StringProperty()
    last_stem: StringProperty()
    last_log_path: StringProperty()
    last_ply_path: StringProperty()


def register():
    bpy.utils.register_class(VipeProperties)


def unregister():
    bpy.utils.unregister_class(VipeProperties)
