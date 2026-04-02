from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Prediction:
    ball_pred: float
    rotor_pred: float | None
    impact_angle: float
    confidence: float


class AlexBotPhysics:
    """Motor físico simplificado para estimar impacto de bola en ruleta."""

    def __init__(self):
        self.ball_history: list[tuple[float, float]] = []
        self.rotor_history: list[tuple[float, float]] = []
        self.max_history = 24
        self.drop_omega_threshold = 3.2

    def update(self, ball_angle: float | None, rotor_angle: float | None = None):
        now = time.time()
        if ball_angle is not None:
            self.ball_history.append((now, ball_angle))
        if rotor_angle is not None:
            self.rotor_history.append((now, rotor_angle))
        self._trim()

    def _trim(self):
        if len(self.ball_history) > self.max_history:
            self.ball_history = self.ball_history[-self.max_history :]
        if len(self.rotor_history) > self.max_history:
            self.rotor_history = self.rotor_history[-self.max_history :]

    @staticmethod
    def _angle_delta(a1: float, a2: float) -> float:
        return ((a2 - a1 + 180.0) % 360.0) - 180.0

    @classmethod
    def _estimate_kinematics(cls, history: list[tuple[float, float]]):
        if len(history) < 5:
            return None, None

        omegas: list[float] = []
        omega_times: list[float] = []
        for i in range(1, len(history)):
            t0, a0 = history[i - 1]
            t1, a1 = history[i]
            dt = t1 - t0
            if dt <= 1e-6:
                continue
            dtheta = cls._angle_delta(a0, a1)
            omegas.append(dtheta / dt)
            omega_times.append((t0 + t1) * 0.5)

        if len(omegas) < 4:
            return None, None

        omega = sum(omegas[-4:]) / min(4, len(omegas))

        alpha_samples: list[float] = []
        for i in range(1, len(omegas)):
            dt = omega_times[i] - omega_times[i - 1]
            if dt <= 1e-6:
                continue
            alpha_samples.append((omegas[i] - omegas[i - 1]) / dt)

        if len(alpha_samples) < 2:
            return omega, None

        alpha = sum(alpha_samples[-4:]) / min(4, len(alpha_samples))
        return omega, alpha

    def get_prediction(self) -> Prediction | None:
        ball_omega, ball_alpha = self._estimate_kinematics(self.ball_history)
        if ball_omega is None or ball_alpha is None:
            return None
        if abs(ball_omega) < self.drop_omega_threshold:
            return None

        t_drop = max(0.25, min(2.5, abs(ball_omega / max(abs(ball_alpha), 1e-4)) * 0.12))

        ball_now = self.ball_history[-1][1]
        impact_angle = (ball_now + ball_omega * t_drop + 0.5 * ball_alpha * (t_drop**2)) % 360

        rotor_pred = None
        rotor_omega, _ = self._estimate_kinematics(self.rotor_history)
        if rotor_omega is not None and self.rotor_history:
            rotor_now = self.rotor_history[-1][1]
            rotor_pred = (rotor_now + rotor_omega * t_drop) % 360

        spread = min(14.0, abs(ball_omega) * 0.05)
        ball_pred = (impact_angle + spread) % 360
        confidence = max(0.0, min(1.0, (abs(ball_omega) / 35.0 + abs(ball_alpha) / 120.0) / 2.0))
        return Prediction(ball_pred=ball_pred, rotor_pred=rotor_pred, impact_angle=impact_angle, confidence=confidence)


class UniversalCylinderMap:
    def __init__(self, mode: str = 'European'):
        self.mode = mode
        self.euro_wheel = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
        self.usa_wheel = [0, 28, 9, 26, 30, 11, 7, 20, 32, 17, 5, 22, 34, 15, 3, 24, 36, 13, 1, '00', 27, 10, 25, 29, 12, 8, 19, 31, 18, 6, 21, 33, 16, 4, 23, 35, 14, 2]

    def set_mode(self, mode: str):
        if mode in {'European', 'American'}:
            self.mode = mode

    @property
    def wheel(self):
        return self.euro_wheel if self.mode == 'European' else self.usa_wheel

    def get_neighbors(self, number: int | str, span: int = 2) -> list[int | str]:
        wheel = self.wheel
        if number not in wheel:
            return []
        idx = wheel.index(number)
        return [wheel[(idx + i) % len(wheel)] for i in range(-span, span + 1)]


class CylinderPhysics:
    """Mapeo de rueda y detección de tendencia por sectores para modo EU/USA."""

    def __init__(self, mode: str = 'European'):
        self.map = UniversalCylinderMap(mode=mode)
        self.sectors = {
            'Voisins': [22, 18, 29, 7, 28, 12, 35, 3, 26, 0, 32, 15, 19, 4, 21, 2, 25],
            'Orphelins': [1, 20, 14, 31, 9, 17, 34, 6],
            'Tier': [33, 16, 24, 5, 10, 23, 8, 30, 11, 36, 13, 27],
        }

    def set_mode(self, mode: str):
        self.map.set_mode(mode)

    def get_sector(self, number: int) -> str:
        for sector, numbers in self.sectors.items():
            if number in numbers:
                return sector
        return 'Unknown'

    def predict_physical_zone(self, last_numbers: list[int]) -> str | None:
        if not last_numbers:
            return None
        recent = last_numbers[-5:]
        sector_count: dict[str, int] = {}
        for number in recent:
            sector = self.get_sector(number)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        best_sector, hits = max(sector_count.items(), key=lambda item: item[1])
        if best_sector != 'Unknown' and hits >= 3:
            return best_sector
        return None
