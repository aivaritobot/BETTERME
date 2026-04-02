from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

try:
    from scipy.integrate import odeint
except Exception:  # pragma: no cover
    def odeint(func, y0, t):
        y = np.zeros((len(t), len(y0)), dtype=float)
        y[0] = y0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            y[i] = y[i - 1] + dt * np.array([func(y[i - 1][0], t[i - 1])])
        return y


@dataclass
class Prediction:
    ball_pred: float
    rotor_pred: float | None
    impact_angle: float
    confidence: float


@dataclass
class BetSuggestion:
    should_bet: bool
    confidence: float
    sector_numbers: list[int | str]
    message: str
    bet_type: str
    amount: float


class AlexBotPhysics:
    """Compatibilidad con tests legacy + predicción cinemática base."""

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
        self.ball_history = self.ball_history[-self.max_history :]
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

        omega = sum(omegas[-4:]) / 4.0
        alpha_samples: list[float] = []
        for i in range(1, len(omegas)):
            dt = omega_times[i] - omega_times[i - 1]
            if dt <= 1e-6:
                continue
            alpha_samples.append((omegas[i] - omegas[i - 1]) / dt)

        alpha = sum(alpha_samples[-4:]) / max(1, min(4, len(alpha_samples)))
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
    def __init__(self, mode: str = "European"):
        self.mode = mode
        self.euro_wheel = [
            0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8,
            23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12,
            35, 3, 26,
        ]
        self.usa_wheel = [
            0, 28, 9, 26, 30, 11, 7, 20, 32, 17, 5, 22, 34, 15, 3, 24, 36,
            13, 1, "00", 27, 10, 25, 29, 12, 8, 19, 31, 18, 6, 21, 33, 16,
            4, 23, 35, 14, 2,
        ]

    def set_mode(self, mode: str):
        if mode in {"European", "American"}:
            self.mode = mode

    @property
    def wheel(self):
        return self.euro_wheel if self.mode == "European" else self.usa_wheel

    def get_neighbors(self, number: int | str, span: int = 2) -> list[int | str]:
        wheel = self.wheel
        if number not in wheel:
            return []
        idx = wheel.index(number)
        return [wheel[(idx + i) % len(wheel)] for i in range(-span, span + 1)]


class CylinderPhysics:
    def __init__(self, mode: str = "European"):
        self.map = UniversalCylinderMap(mode=mode)
        self.sectors = {
            "Voisins": [22, 18, 29, 7, 28, 12, 35, 3, 26, 0, 32, 15, 19, 4, 21, 2, 25],
            "Orphelins": [1, 20, 14, 31, 9, 17, 34, 6],
            "Tier": [33, 16, 24, 5, 10, 23, 8, 30, 11, 36, 13, 27],
        }

    def set_mode(self, mode: str):
        self.map.set_mode(mode)

    def get_sector(self, number: int) -> str:
        for sector, numbers in self.sectors.items():
            if number in numbers:
                return sector
        return "Unknown"

    def predict_physical_zone(self, last_numbers: list[int]) -> str | None:
        if not last_numbers:
            return None
        recent = last_numbers[-5:]
        counts: dict[str, int] = {}
        for n in recent:
            sector = self.get_sector(n)
            counts[sector] = counts.get(sector, 0) + 1
        best_sector, hits = max(counts.items(), key=lambda i: i[1])
        return best_sector if best_sector != "Unknown" and hits >= 3 else None


class RoulettePhysicsEngine:
    """Modelo mejorado: fricción lineal + Coulomb + integración + dispersión gaussiana."""

    def __init__(self):
        self.cylinder = UniversalCylinderMap(mode="European")
        self.k_linear = 0.16
        self.k_coulomb = 2.0
        self.drop_omega = 12.0
        self.dispersion_deg = 10.0
        self.spin_errors: list[float] = []

    @staticmethod
    def _angle_delta(a1: float, a2: float) -> float:
        return ((a2 - a1 + 180.0) % 360.0) - 180.0

    def fit_friction(self, angle_history: list[tuple[float, float]]) -> None:
        if len(angle_history) < 8:
            return
        omegas = []
        alphas = []
        for i in range(1, len(angle_history) - 1):
            t0, a0 = angle_history[i - 1]
            t1, a1 = angle_history[i]
            t2, a2 = angle_history[i + 1]
            dt1 = max(1e-4, t1 - t0)
            dt2 = max(1e-4, t2 - t1)
            w1 = self._angle_delta(a0, a1) / dt1
            w2 = self._angle_delta(a1, a2) / dt2
            alpha = (w2 - w1) / max(1e-4, (t2 - t0) * 0.5)
            omegas.append(w1)
            alphas.append(alpha)

        if len(omegas) < 5:
            return

        x1 = np.array(omegas)
        x2 = np.sign(x1)
        y = -np.array(alphas)
        X = np.column_stack([x1, x2])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.k_linear = float(np.clip(coef[0], 0.01, 1.0))
        self.k_coulomb = float(np.clip(coef[1], 0.1, 30.0))

    def learn_dispersion(self, errors_deg: list[float]) -> None:
        self.spin_errors.extend(errors_deg)
        self.spin_errors = self.spin_errors[-30:]
        if len(self.spin_errors) >= 3:
            self.dispersion_deg = float(np.clip(np.std(self.spin_errors), 5.0, 22.0))

    def _omega_rhs(self, omega: float, _t: float) -> float:
        return -self.k_linear * omega - self.k_coulomb * np.sign(omega)

    def predict_drop(self, now_angle: float, now_omega: float, rotor_angle: float | None, rotor_omega: float | None):
        t = np.linspace(0, 6.0, 300)
        omega_series = odeint(lambda w, tt: self._omega_rhs(float(w), tt), y0=[now_omega], t=t).flatten()

        idx = np.where(np.abs(omega_series) <= self.drop_omega)[0]
        stop_idx = int(idx[0]) if len(idx) else len(t) - 1
        t_drop = float(t[stop_idx])

        theta = now_angle
        for i in range(1, stop_idx + 1):
            dt = t[i] - t[i - 1]
            theta = (theta + omega_series[i - 1] * dt) % 360.0

        relative = theta
        if rotor_angle is not None and rotor_omega is not None:
            rotor_pred = (rotor_angle + rotor_omega * t_drop) % 360.0
            relative = (theta - rotor_pred) % 360.0

        bounce_noise = np.random.uniform(-10.0, 10.0)
        impact = (relative + bounce_noise) % 360.0
        return impact, t_drop

    def sector_from_angle(self, angle: float, span_numbers: int = 10) -> list[int | str]:
        wheel = self.cylinder.wheel
        idx = int((angle / 360.0) * len(wheel)) % len(wheel)
        half = span_numbers // 2
        return [wheel[(idx + i) % len(wheel)] for i in range(-half, half + 1)]

    def confidence_and_span(self) -> tuple[float, int]:
        conf = float(np.clip(1.0 - self.dispersion_deg / 30.0, 0.45, 0.95))
        span = int(np.clip(round(12 - (conf - 0.5) * 8), 8, 12))
        return conf, span

    def suggest_bet(self, bankroll: float, sector: list[int | str], confidence: float) -> BetSuggestion:
        should_bet = confidence >= 0.70
        base = max(1.0, bankroll * 0.01)
        amount = float(round(base if should_bet else 0.0, 2))

        # selección simple por edge aproximado
        if confidence > 0.83 and len(sector) <= 9:
            bet_type = "straight-up (35:1)"
            target = sector[len(sector) // 2]
            msg = f"Apuesta {amount} al {target}"
        elif confidence > 0.75:
            bet_type = "split (17:1)"
            msg = f"Apuesta {amount} split en sector {sector[0]}-{sector[-1]}"
        else:
            bet_type = "docena (2:1)"
            nums = [n for n in sector if isinstance(n, int)]
            dozen = (min(nums) - 1) // 12 + 1 if nums else 2
            msg = f"Apuesta {amount} a docena {dozen} / sector {sector[0]}-{sector[-1]}"

        return BetSuggestion(
            should_bet=should_bet,
            confidence=confidence,
            sector_numbers=sector,
            message=msg,
            bet_type=bet_type,
            amount=amount,
        )
