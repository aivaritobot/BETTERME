from __future__ import annotations

import argparse

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Genera un video demo sintético para pruebas locales')
    parser.add_argument('--output', default='demo.mp4')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    return parser


def main(args: argparse.Namespace) -> int:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))

    cx, cy = args.width // 2, args.height // 2
    for i in range(args.frames):
        frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        ball_ang = np.deg2rad((i * 8) % 360)
        rotor_ang = np.deg2rad((i * 2) % 360)

        bx = int(cx + 0.35 * args.width * np.cos(ball_ang))
        by = int(cy + 0.35 * args.height * np.sin(ball_ang))
        rx = int(cx + 0.20 * args.width * np.cos(rotor_ang))
        ry = int(cy + 0.20 * args.height * np.sin(rotor_ang))

        cv2.circle(frame, (bx, by), 8, (255, 255, 255), -1)
        cv2.circle(frame, (rx, ry), 8, (0, 255, 0), -1)
        cv2.circle(frame, (cx, cy), 250, (50, 50, 50), 2)

        out.write(frame)

    out.release()
    print(f'Video demo generado: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(build_parser().parse_args()))
