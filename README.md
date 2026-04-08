# ViPE Blender Addon

A Blender addon that runs [NVIDIA ViPE](https://github.com/NVlabs/vipe) (Video Pose Engine) from inside Blender and automatically imports the resulting camera trajectory and point cloud into the scene.

## What it does

1. Launches `vipe infer` as a background subprocess from the Blender N-Panel
2. Monitors progress and imports results when done:
   - **Camera trajectory** → animated Blender camera with correct intrinsics
   - **Point cloud** → vertex-coloured mesh parented to the same world space
3. Handles coordinate system conversion from ViPE/OpenCV convention to Blender convention

## Requirements

- Blender 4.0+
- A working [ViPE](https://github.com/NVlabs/vipe) installation in a conda environment
- The ViPE repo cloned locally (the addon calls `python run.py` directly)

## Installation

1. Download or clone this repo and zip the folder:
   ```bash
   zip -r vipe_blender.zip vipe_blender/
   ```
2. In Blender: **Edit → Preferences → Add-ons → Install** → select `vipe_blender.zip`
3. Enable **ViPE Camera Tracker** in the add-ons list

The panel appears in the **N-Panel** (press `N` in the 3D Viewport) under the **ViPE** tab.

## Usage

### Paths
| Field | Description |
|---|---|
| ViPE Directory | Root of the ViPE repository (contains `run.py`) |
| Conda Env | Name of the conda environment with ViPE installed |
| Input | Video file (`.mp4`, `.mov`, `.avi`, …) or image directory |
| Output Dir | Where ViPE saves results (poses, depth, point cloud) |

### Intrinsics
Choose how camera intrinsics are estimated:

| Mode | Description |
|---|---|
| **GeoCalib (auto)** | AI-powered calibration — samples a few frames (recommended) |
| **Manual FOV** | Specify vertical field-of-view in degrees |
| **Manual (fx fy cx cy)** | Provide intrinsics directly in pixels |
| **Calibration File** | Read `fx fy cx cy` from a text file (DROID-SLAM format) |

### Pipeline
Select a named ViPE pipeline preset. Expand **Overrides** to control depth model, alignment, and output options individually.

### Frame Range
Expand to limit which frames are processed (`Frame Start`, `Frame End`, `Frame Skip`). Default processes all frames. For long videos, setting **Frame Skip = 2** halves memory usage.

### Running
Click **Run ViPE**. Status updates in the panel. Once done, the camera and point cloud are imported automatically. Use **Open Log** to inspect the full output if something goes wrong.

The **Import** section at the bottom lets you manually re-import the camera or point cloud from a previous run.

## Output files

ViPE saves results to the output directory with the following structure:

```
output_dir/
  pose/          ← camera-to-world 4×4 matrices (.npz)
  intrinsics/    ← fx, fy, cx, cy per frame (.npz)
  depth/         ← metric depth maps
  mask/          ← instance segmentation masks
  vipe/          ← SLAM map (.pt) and visualisation video
```

The addon also writes `{stem}_slam_map.ply` alongside the `.pt` file for point cloud import.

## Notes

- The addon modifies two files in the ViPE repo to add support for **manual intrinsics** (`pipeline.init.intrinsics=manual`) and **non-mp4 video formats**. These changes are minimal and non-breaking.
- Coordinate system: ViPE uses OpenCV convention (Y down, Z forward). The addon compensates with a per-pose rotation flip and a parent empty at −90° X.
