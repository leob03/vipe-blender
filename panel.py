import bpy


class VIPE_PT_Panel(bpy.types.Panel):
    bl_label       = "ViPE Camera Tracker"
    bl_idname      = "VIPE_PT_panel"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'ViPE'

    def draw(self, context):
        layout = self.layout
        props  = context.scene.vipe

        from . import operators as _ops
        running = (_ops._process is not None and _ops._process.poll() is None)

        # ------------------------------------------------------------ Paths
        box = layout.box()
        box.label(text="Paths", icon='FILE_FOLDER')
        box.prop(props, "vipe_dir")
        box.prop(props, "conda_env")
        box.prop(props, "input_type")
        if props.input_type == 'VIDEO':
            box.prop(props, "input_path", text="Video File")
        else:
            box.prop(props, "input_path", text="Image Directory")
        box.prop(props, "output_dir")

        # --------------------------------------------------------- Intrinsics
        box = layout.box()
        box.label(text="Intrinsics", icon='CAMERA_DATA')
        box.prop(props, "intrinsics_mode", text="Mode")
        if props.intrinsics_mode == 'fov':
            box.prop(props, "fov_y_deg")
        elif props.intrinsics_mode == 'manual':
            row = box.row(align=True)
            row.prop(props, "fx")
            row.prop(props, "fy")
            row = box.row(align=True)
            row.prop(props, "cx")
            row.prop(props, "cy")
        elif props.intrinsics_mode == 'calib':
            box.prop(props, "calib_file")

        # --------------------------------------------------------- Pipeline
        box = layout.box()
        box.label(text="Pipeline", icon='SETTINGS')
        box.prop(props, "pipeline_preset")

        row = box.row()
        row.prop(props, "show_overrides",
                 icon='TRIA_DOWN' if props.show_overrides else 'TRIA_RIGHT',
                 emboss=False)
        if props.show_overrides:
            sub = box.column(align=True)
            sub.prop(props, "camera_type")
            sub.separator()
            sub.prop(props, "depth_model")
            sub.prop(props, "depth_align_model")
            sub.separator()
            sub.prop(props, "save_slam_map")
            sub.prop(props, "save_viz")

        # ------------------------------------------------------ Frame Range (collapsed by default)
        box = layout.box()
        row = box.row()
        row.prop(props, "show_frame_range",
                 icon='TRIA_DOWN' if props.show_frame_range else 'TRIA_RIGHT',
                 emboss=False)
        if props.show_frame_range:
            sub = box.column(align=True)
            row2 = sub.row(align=True)
            row2.prop(props, "frame_start")
            row2.prop(props, "frame_end")
            sub.prop(props, "frame_skip")

        # ------------------------------------------------------- Run / status
        box = layout.box()
        box.label(text="Run", icon='PLAY')

        row = box.row(align=True)
        row.operator("vipe.run",    text="Run ViPE", icon='PLAY')
        row.operator("vipe.cancel", text="",         icon='X')

        box.label(text=props.status)

        if props.last_log_path:
            box.operator("vipe.open_log", icon='TEXT')

        # ------------------------------------------------------- Import
        box = layout.box()
        box.label(text="Import", icon='IMPORT')
        box.prop(props, "blender_start_frame")
        box.separator()

        col = box.column(align=True)
        col.prop(props, "import_pointcloud")
        if props.import_pointcloud:
            sub = col.column(align=True)
            sub.prop(props, "pointcloud_mode", text="")
            if props.pointcloud_mode == 'PER_FRAME_DENSE':
                sub.prop(props, "depth_pcd_stride")
            else:
                sub.enabled = props.save_slam_map
            sub.prop(props, "colored_pointcloud")
            row = sub.row(align=True)
            row.prop(props, "point_radius")
            row.operator("vipe.set_point_radius", text="", icon='FILE_REFRESH')
        box.separator()

        row = box.row(align=True)
        row.operator("vipe.import_camera",     icon='CAMERA_DATA')
        row.operator("vipe.import_pointcloud", icon='POINTCLOUD_DATA')


CLASSES = [VIPE_PT_Panel]
