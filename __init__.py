bl_info = {
    "name": "ViPE Camera Tracker",
    "author": "leob03",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "Properties > Scene > ViPE Camera Tracker",
    "description": "Run NVIDIA ViPE from Blender and import camera trajectory + point cloud",
    "category": "Camera",
}

from . import properties, operators, panel


def register():
    import bpy
    from bpy.utils import register_class

    properties.register()

    for cls in operators.CLASSES:
        register_class(cls)
    for cls in panel.CLASSES:
        register_class(cls)

    bpy.types.Scene.vipe = bpy.props.PointerProperty(type=properties.VipeProperties)


def unregister():
    import bpy
    from bpy.utils import unregister_class

    for cls in reversed(panel.CLASSES):
        unregister_class(cls)
    for cls in reversed(operators.CLASSES):
        unregister_class(cls)

    properties.unregister()

    del bpy.types.Scene.vipe


if __name__ == "__main__":
    register()
