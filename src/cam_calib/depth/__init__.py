"""Optional depth providers.

Currently:
  - ``foundation_stereo``: HTTP client for the FoundationStereo inference
    server (https://github.com/williamshen-nz/FoundationStereo). Use it to
    replace hardware depth with learned stereo depth in the visualize flow.
"""
from cam_calib.depth.foundation_stereo import (
    FoundationStereoClient,
    DEFAULT_FS_URL,
    warp_depth_ir_to_color,
)

__all__ = [
    "FoundationStereoClient",
    "DEFAULT_FS_URL",
    "warp_depth_ir_to_color",
]
