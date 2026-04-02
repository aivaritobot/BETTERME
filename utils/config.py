from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ROI:
    top: int
    left: int
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ROI':
        return cls(
            top=int(data['top']),
            left=int(data['left']),
            width=int(data['width']),
            height=int(data['height']),
        )


@dataclass(frozen=True)
class DetectionConfig:
    ball_sat_max: int = 70
    ball_block_size: int = 21
    ball_c_offset: int = -12
    ball_min_area: int = 8
    rotor_hsv_lower: tuple[int, int, int] = (35, 70, 60)
    rotor_hsv_upper: tuple[int, int, int] = (95, 255, 255)
    rotor_min_area: int = 15


@dataclass(frozen=True)
class PhysicsConfig:
    # Ventana temporal
    max_history: int = 32
    fit_window: int = 12

    # Umbral y horizonte
    drop_omega_threshold: float = 3.2
    min_drop_time: float = 0.2
    max_drop_time: float = 8.0

    # Integrador
    integrator_dt: float = 0.01

    # Fricción bola
    min_linear_drag: float = 0.01
    coulomb_drag: float = 0.35

    # Rotor (más estable que bola)
    rotor_linear_drag: float = 0.02
    rotor_coulomb_drag: float = 0.08

    # Deflectores / dispersión
    spread_gain: float = 0.12
    spread_bias: float = 0.4
    max_deflector_spread_deg: float = 18.0


def load_roi_from_config(path: str = 'config.json') -> ROI:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f'No existe {path}. Ejecuta calibrate.py o usa --roi-manual.'
        )

    with config_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return ROI.from_dict(data['roi'])


def parse_manual_roi(raw: str) -> ROI:
    try:
        top, left, width, height = [int(x.strip()) for x in raw.split(',')]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            'Formato ROI inválido. Usa: top,left,width,height'
        ) from exc

    if width <= 0 or height <= 0:
        raise ValueError('ROI inválida: width y height deben ser > 0')

    return ROI(top=top, left=left, width=width, height=height)
