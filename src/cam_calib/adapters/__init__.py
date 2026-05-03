"""Camera-capture adapters.

Each adapter implements ``CameraSource`` so workflows can stay backend-agnostic.
RealSense support requires the optional ``[realsense]`` extra (pyrealsense2).
"""
from cam_calib.adapters.base import CameraSource, CameraFrame, StereoIRData

__all__ = ["CameraSource", "CameraFrame", "StereoIRData"]
