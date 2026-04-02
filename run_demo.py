from __future__ import annotations

import argparse

from main import build_parser, run


def demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Atajo de arranque para demos locales')
    parser.add_argument('--mode', choices=['video', 'screen'], default='video')
    parser.add_argument('--video-path', default='demo.mp4')
    parser.add_argument('--roi-manual', default='0,0,800,800')
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--max-frames', type=int, default=0)
    return parser


def main() -> int:
    args = demo_parser().parse_args()

    cli = build_parser().parse_args([])
    cli.source = args.mode
    cli.video_path = args.video_path if args.mode == 'video' else None
    cli.roi_manual = args.roi_manual
    cli.log_level = args.log_level
    cli.max_frames = args.max_frames
    cli.config = 'config.json'
    return run(cli)


if __name__ == '__main__':
    raise SystemExit(main())
