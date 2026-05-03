"""``python -m cam_calib <subcommand>`` — entry point.

Subcommands:
    calibrate     Live ChArUco extrinsic loop on RealSense cameras
    list-cameras  Print connected RealSense serials
    show-board    Print the path to the bundled ChArUco board PDF
"""
import argparse
import sys
from importlib import resources
from pathlib import Path


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from cam_calib.adapters.realsense import SimpleRealSense
    from cam_calib.workflows.calibrate_extrinsics import run_calibration_loop

    if args.cameras:
        serials = [s.strip() for s in args.cameras.split(",") if s.strip()]
    else:
        serials = SimpleRealSense.list_connected_serials()
        if not serials:
            print("no RealSense cameras detected", file=sys.stderr)
            return 1
        print(f"auto-detected: {serials}")

    extrinsics_dir = Path(args.extrinsics_dir).expanduser().resolve()
    print(f"writing extrinsics → {extrinsics_dir}")

    cams = []
    try:
        for s in serials:
            cam = SimpleRealSense(s, resolution=args.resolution, fps=args.fps)
            cam.start()
            cams.append(cam)
        run_calibration_loop(cams, extrinsics_dir, visualize=not args.no_visualize)
    finally:
        for cam in cams:
            cam.stop()
    return 0


def _cmd_list_cameras(_: argparse.Namespace) -> int:
    from cam_calib.adapters.realsense import SimpleRealSense

    serials = SimpleRealSense.list_connected_serials()
    if not serials:
        print("no RealSense cameras detected", file=sys.stderr)
        return 1
    for s in serials:
        print(s)
    return 0


def _cmd_show_board(_: argparse.Namespace) -> int:
    pdf = resources.files("cam_calib.assets").joinpath("charuco_board.pdf")
    print(str(pdf))
    return 0


def _parse_resolution(value: str) -> tuple:
    try:
        w, h = value.lower().split("x")
        return (int(w), int(h))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid WxH '{value}'") from e


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cam-calib")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Live ChArUco extrinsic loop")
    p_cal.add_argument("--cameras", default=None,
                       help="comma-separated RealSense serials (default: auto)")
    p_cal.add_argument("--extrinsics-dir", default="./cam_extrinsics",
                       help="output directory for <serial>.yaml files")
    p_cal.add_argument("--resolution", type=_parse_resolution, default=(1280, 720),
                       help="capture resolution WxH (default 1280x720)")
    p_cal.add_argument("--fps", type=int, default=30)
    p_cal.add_argument("--no-visualize", action="store_true")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_list = sub.add_parser("list-cameras", help="Print connected RealSense serials")
    p_list.set_defaults(func=_cmd_list_cameras)

    p_board = sub.add_parser("show-board", help="Print bundled ChArUco PDF path")
    p_board.set_defaults(func=_cmd_show_board)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
