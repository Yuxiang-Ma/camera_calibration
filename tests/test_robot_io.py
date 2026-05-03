import numpy as np

from cam_calib.core.robot_io import (
    load_robot_base_pose,
    robot_base_pose_filename,
    save_robot_base_pose,
)


def test_filename_with_arm():
    assert (
        robot_base_pose_filename("franka", "adelson", arm="left")
        == "left_franka_adelson_base_pose_in_world.yaml"
    )


def test_filename_without_arm():
    assert (
        robot_base_pose_filename("franka", "adelson")
        == "franka_adelson_base_pose_in_world.yaml"
    )


def test_save_load_round_trip(tmp_path):
    T = np.eye(4)
    T[:3, 3] = [-0.395, -0.62, 0.0]
    save_robot_base_pose(T, tmp_path, robot="franka", lab="adelson", arm="left")
    loaded = load_robot_base_pose(tmp_path, robot="franka", lab="adelson", arm="left")
    np.testing.assert_allclose(loaded, T)
    # Other arm absent
    assert load_robot_base_pose(tmp_path, robot="franka", lab="adelson", arm="right") is None
