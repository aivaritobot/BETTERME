from __future__ import annotations

import argparse
import json
import logging

import cv2
import mss
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Calibración automática de ROI de ruleta en pantalla')
    parser.add_argument('--output', default='config.json', help='Archivo de salida de configuración')
    parser.add_argument('--min-radius', type=int, default=100)
    parser.add_argument('--max-radius', type=int, default=400)
    return parser


def auto_calibrate(min_radius: int = 100, max_radius: int = 400):
    sct = mss.mss()
    monitor = sct.monitors[1]
    img = np.array(sct.grab(monitor))
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.uint16(np.around(circles))
    best = circles[0, 0]

    cx, cy, radius = int(best[0]), int(best[1]), int(best[2])
    config = {
        'center_x': cx,
        'center_y': cy,
        'radius': radius,
        'roi': {
            'top': max(0, cy - radius),
            'left': max(0, cx - radius),
            'width': radius * 2,
            'height': radius * 2,
        },
    }
    return config


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger('calibrate')

    logger.info('Buscando ruleta en pantalla...')
    config = auto_calibrate(min_radius=args.min_radius, max_radius=args.max_radius)
    if config is None:
        logger.error('No se encontró ruleta. Ajusta rango de radios o visibilidad de la mesa.')
        return 1

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    logger.info('Calibración exitosa. Config guardada en %s', args.output)
    logger.info('Centro=%s,%s radio=%s', config['center_x'], config['center_y'], config['radius'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main(build_parser().parse_args()))
