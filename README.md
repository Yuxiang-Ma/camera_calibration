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

```bash
# Core only (no hardware, no viz)
pip install -e ~/src/cam_calib

# With RealSense capture
pip install -e ~/src/cam_calib[realsense]

# Full convenience: RealSense + viz + URDF mesh sampling
pip install -e ~/src/cam_calib[all]
```

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
