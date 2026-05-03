import numpy as np

from cam_calib.core.geometry import (
    DEFAULT_T_world_board,
    invert_se3,
    se3_from_rvec_tvec,
)


def test_invert_se3_round_trip():
    rng = np.random.default_rng(0)
    # Random rotation via QR + a small translation
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = rng.normal(size=3)
    T_inv = invert_se3(T)
    np.testing.assert_allclose(T_inv @ T, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-10)


def test_default_T_world_board_z_flip():
    # A point on the board (positive board-Z) should land at negative world-Z.
    p_board = np.array([0.0, 0.0, 0.05, 1.0])
    p_world = np.linalg.inv(DEFAULT_T_world_board) @ p_board
    assert p_world[2] < 0


def test_se3_from_rvec_tvec_identity():
    T = se3_from_rvec_tvec(np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_se3_from_rvec_tvec_translation():
    T = se3_from_rvec_tvec(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
