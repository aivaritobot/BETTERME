from __future__ import annotations

import argparse
import os
import time

from engine.bankroll import RiskManager
from engine.physics import CylinderPhysics
from engine.statistics import RouletteAuditor
from engine.vision import RouletteVision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ALEXBOT V2: visión + física + banca')
    parser.add_argument('--window-size', type=int, default=100, help='Tamaño de ventana estadística')
    parser.add_argument('--capital', type=float, default=100.0, help='Capital inicial')
    parser.add_argument('--stop-loss', type=float, default=30.0, help='Límite de pérdida')
    parser.add_argument('--take-profit', type=float, default=150.0, help='Objetivo de ganancia')
    parser.add_argument('--scan-interval', type=float, default=1.0, help='Intervalo de escaneo en segundos')
    parser.add_argument('--region', default='500,200,100,100', help='Región OCR: x,y,w,h')
    return parser


def _parse_region(raw: str) -> tuple[int, int, int, int]:
    x, y, w, h = (int(part.strip()) for part in raw.split(','))
    if w <= 0 or h <= 0:
        raise ValueError('w y h deben ser mayores a 0')
    return x, y, w, h


def start_alexbot(args: argparse.Namespace) -> int:
    region = _parse_region(args.region)
    vision = RouletteVision(region=region)
    physics = CylinderPhysics()
    stats = RouletteAuditor(window_size=args.window_size)
    risk = RiskManager(
        initial_capital=args.capital,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )

    print('ALEXBOT V2: SISTEMA DE VISIÓN Y FÍSICA ACTIVADO')
    print(f'Región OCR: {region}')

    while risk.session_active:
        new_number = vision.get_last_number()
        betting_open = vision.is_betting_open()

        if new_number is not None:
            print(f'Número detectado: {new_number}')
            stats.add_number(new_number)

            sector = physics.get_sector(new_number)
            physical_trend = physics.predict_physical_zone(list(stats.history))
            prob_gaps = stats.get_probability_gap()

            print(f'Sector actual: {sector}')
            print(f'Gap estadístico: {prob_gaps}')

            if physical_trend:
                print(f'🔥 ALERTA FÍSICA: tendencia en {physical_trend}')

            if betting_open is False:
                print('⛔ NO MORE BETS detectado (semáforo rojo).')

        time.sleep(args.scan_interval)

    return 0


def run(args: argparse.Namespace) -> int:
    return start_alexbot(args)


if __name__ == '__main__':
    if os.environ.get('PYTHONUNBUFFERED') is None:
        os.environ['PYTHONUNBUFFERED'] = '1'
    raise SystemExit(run(build_parser().parse_args()))
