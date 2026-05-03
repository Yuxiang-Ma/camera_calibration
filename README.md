# cam_calib

Standalone camera extrinsic calibration package using a ChArUco board.

Originally extracted from `tac_foundation/ReKep/vision/calibration/`. Designed
to be **camera-agnostic at the core**, with optional adapters for live capture
and visualization. **No robot-SDK dependencies** — joint angles and base poses
are passed in by the caller.

## Layout

```
src/cam_calib/
├── core/         # pure CV math + YAML I/O (numpy in, numpy out)
├── adapters/     # camera I/O backends (RealSense via [realsense] extra)
├── workflows/    # end-user entry points; compose core + adapter + viz
├── viz/          # Open3D / Rerun helpers (optional via [viz] extra)
├── robot/        # URDF FK + mesh sampling for overlay viz (optional via [robot-viz])
└── assets/       # ChArUco board PDF
```

Dependency direction is strict: `workflows → core, adapters, viz`. Core never
imports adapters or viz.

## Install

The package itself only depends on `numpy<2` and `pyyaml` — extras are opt-in
because they tend to fight other packages in shared envs.

```bash
# Core only (assumes you already have OpenCV with the aruco contrib module)
pip install -e ~/src/camera_calibration

# Add OpenCV (only if you don't already have opencv-contrib-python)
pip install -e ~/src/camera_calibration[opencv]

# RealSense capture
pip install -e ~/src/camera_calibration[realsense]

# Visualization (Open3D + Rerun)
pip install -e ~/src/camera_calibration[viz]

# URDF mesh sampling (yourdfpy + trimesh)
pip install -e ~/src/camera_calibration[robot-viz]
```

> **Why no `[all]` extra?** Combining viz + robot-viz can pull `numpy>=2` or
> upgrade Open3D in ways that break neighboring packages (`gs-sdk`,
> `normalflow`, `openpi-client`, etc.). Install only the extras you need.

> **OpenCV note.** `opencv-python` and `opencv-contrib-python` are mutually
> exclusive distributions (both ship the `cv2` namespace). We deliberately do
> not list either as a hard dependency. If your env already has one of them
> with the `cv2.aruco` module, you're fine — `cam_calib` checks at import.

## Quick start

```python
import cv2
from cam_calib.core import (
    DEFAULT_BOARD, detect_board, estimate_board_pose,
    save_cam_extrinsics,
)
from cam_calib.core.geometry import DEFAULT_T_world_board

img = cv2.imread("frame.png")
K = ...        # (3, 3) intrinsics
dist = ...     # (5,) distortion coefficients

det = detect_board(img, K, dist, DEFAULT_BOARD)
pose = estimate_board_pose(det, K, dist, DEFAULT_BOARD)

# board pose is in camera frame; world origin = board origin (Z-down convention)
T_cam_world = pose.T_board_cam @ DEFAULT_T_world_board
save_cam_extrinsics("123456789", T_cam_world, "data/cam_extrinsics")
```

## CLI

```bash
# Live ChArUco loop on all connected RealSense cameras
python -m cam_calib calibrate

# Specific cameras
python -m cam_calib calibrate --cameras 138422070384,134322071848

# Visualize fused point cloud (uses saved extrinsics)
python -m cam_calib visualize
```

### FoundationStereo for visualize depth

`visualize` will try the [FoundationStereo](https://github.com/williamshen-nz/FoundationStereo)
inference server at `http://localhost:1234` by default. If it's reachable,
cameras are opened with IR stereo streams enabled and FS depth replaces
hardware depth (better far-field accuracy). If it isn't reachable, hardware
depth is used and the IR streams aren't enabled (saves USB bandwidth).

```bash
# Default: try http://localhost:1234, fall back to hardware depth
python -m cam_calib visualize

# Explicit URL
python -m cam_calib visualize --foundation-stereo-url http://my-fs-host:1234

# Skip FS entirely (don't even probe)
python -m cam_calib visualize --no-foundation-stereo
```

When the server is reachable but a per-frame inference fails (timeout, server
error, etc.), `cam_calib` warns and falls back to that camera's hardware depth
for that frame.

## Robot-side

The package itself never imports a robot SDK. If you want to overlay robot
meshes on the fused point cloud, supply a list of pre-computed `(N, 3)` point
clouds in world frame to `workflows.visualize_fused.run`. A convenience helper
`robot.urdf_pcd.urdf_to_pcd(urdf_path, joint_angles, T_base_world)` is provided
under the `[robot-viz]` extra; it uses `yourdfpy` + `trimesh` only.

## YAML schemas

### Camera extrinsics — `<serial>.yaml`

```yaml
matrix:
- [r00, r01, r02, tx]
- [r10, r11, r12, ty]
- [r20, r21, r22, tz]
- [0.0, 0.0, 0.0, 1.0]
shape: [4, 4]
dtype: float64
description: 'World-to-Camera transformation matrix (T_cam_world) for RealSense <serial>. Usage: P_cam = matrix @ P_world.'
```

### Robot base pose — `<arm>_<robot>_<lab>_base_pose_in_world.yaml`

Same structure, different filename convention. `matrix` is `T_base_world`.

## Compatibility shim

The original ReKep callers (`ReKep.vision.drivers.realsense_driver.RealSense.calibrate_extrinsics`,
`ReKep.vision.calibration.aruco`, etc.) continue to work — they forward to this
package.
