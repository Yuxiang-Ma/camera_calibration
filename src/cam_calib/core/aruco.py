"""Single-marker ArUco pose estimation.

Same behavior as the legacy ``ReKep.vision.calibration.aruco.get_marker_pose``,
just with typed return values.
"""
from typing import Optional

import cv2
import numpy as np

from cam_calib.core.types import MarkerPose


def detect_marker_pose(
    image: np.ndarray,
    K: np.ndarray,
    dist: Optional[np.ndarray] = None,
    marker_id: int = 24,
    marker_size: float = 0.094,
    aruco_dict_type: int = cv2.aruco.DICT_5X5_100,
    annotate: bool = True,
) -> tuple:
    """Detect a single ArUco marker by ID and estimate its pose.

    Returns ``(MarkerPose | None, annotated_image)``. The image is always
    returned (annotated if ``annotate`` and the marker was found) so callers
    can show it without an extra branch.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = np.ascontiguousarray(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None:
        return None, image

    for i, detected_id in enumerate(ids.flatten()):
        if int(detected_id) != int(marker_id):
            continue

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[i], marker_size, K, dist
        )
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()

        if annotate:
            cv2.aruco.drawDetectedMarkers(image, [corners[i]])
            cv2.drawFrameAxes(
                image, K, dist if dist is not None else np.zeros((5,)),
                rvec, tvec, marker_size * 0.5,
            )

        return (
            MarkerPose(
                rvec=rvec.flatten(),
                tvec=tvec.flatten(),
                T_marker_cam=T,
            ),
            image,
        )

    return None, image
