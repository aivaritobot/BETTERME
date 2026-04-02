from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

import math
import time
from dataclasses import dataclass

import numpy as np
try:
    from sklearn.linear_model import LinearRegression
except Exception:  # pragma: no cover
    class LinearRegression:  # minimal fallback
        def __init__(self):
            self.coef_ = [0.0, 0.0]
        def fit(self, X, y):
            self.coef_ = [0.0, 0.0]

try:
    from filterpy.kalman import ParticleFilter
except Exception:  # pragma: no cover
    ParticleFilter = None  # type: ignore

try:
    from scipy.integrate import odeint
except Exception:  # pragma: no cover
    odeint = None  # type: ignore


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
    def __init__(self):
        self.ball_history: list[tuple[float, float]] = []
        self.rotor_history: list[tuple[float, float]] = []

    @staticmethod
    def _angle_delta(a1: float, a2: float) -> float:
        return ((a2 - a1 + 180.0) % 360.0) - 180.0

    def get_prediction(self) -> Prediction | None:
        if len(self.ball_history) < 3:
            return None
        (t0, a0), (t1, a1), (t2, a2) = self.ball_history[-3:]
        dt = max(1e-4, t2 - t1)
        w = self._angle_delta(a1, a2) / dt
        a = (self._angle_delta(a0, a1) / max(1e-4, t1 - t0) - w) / max(1e-4, dt)
        t_drop = min(3.0, max(0.2, abs(w / max(abs(a), 1e-3)) * 0.1))
        impact = (a2 + w * t_drop + 0.5 * a * t_drop * t_drop) % 360
        conf = float(np.clip(abs(w) / 100.0, 0.0, 1.0))
        return Prediction(ball_pred=impact, rotor_pred=None, impact_angle=impact, confidence=conf)


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

    @property
    def wheel(self):
        return self.euro_wheel if self.mode == "European" else self.usa_wheel

    def set_mode(self, mode: str):
        if mode in {"European", "American"}:
            self.mode = mode

    def get_neighbors(self, number: int | str, span: int = 2) -> list[int | str]:
        wheel = self.wheel
        if number not in wheel:
            return []
        i = wheel.index(number)
        return [wheel[(i + j) % len(wheel)] for j in range(-span, span + 1)]


class CylinderPhysics:
    def __init__(self, mode: str = "European"):
        self.map = UniversalCylinderMap(mode)
        self.sectors = {
            "Voisins": [22, 18, 29, 7, 28, 12, 35, 3, 26, 0, 32, 15, 19, 4, 21, 2, 25],
            "Orphelins": [1, 20, 14, 31, 9, 17, 34, 6],
            "Tier": [33, 16, 24, 5, 10, 23, 8, 30, 11, 36, 13, 27],
        }

    def get_sector(self, number: int) -> str:
        for k, v in self.sectors.items():
            if number in v:
                return k
        return "Unknown"

    def predict_physical_zone(self, last_numbers: list[int]) -> str | None:
        if not last_numbers:
            return None
        counts: dict[str, int] = {}
        for n in last_numbers[-5:]:
            s = self.get_sector(n)
            counts[s] = counts.get(s, 0) + 1
        name, qty = max(counts.items(), key=lambda x: x[1])
        return name if name != "Unknown" and qty >= 3 else None


class AdvancedPhysicsEngine:
    r"""Modelo híbrido 2D + calibración progresiva para investigación.

    \[ \frac{d^2\theta_b}{dt^2} = -\mu g \operatorname{sign}(\omega_b) - \beta \omega_b^2 - \gamma (\omega_b - \omega_w) \]
    """

    def __init__(self):
        self.mu = 0.015
        self.beta = 0.0009
        self.gamma = 0.04
        self.g = 9.81
        self.dispersion_std_deg = 14.0
        self.factor_tilt = 1.0
        self.edge_threshold = 0.12
        self.confidence_threshold = 0.68
        self.bankroll_fraction = 0.5
        self.cylinder = UniversalCylinderMap(mode="European")
        self.ball_hist: list[tuple[float, float]] = []
        self.rotor_hist: list[tuple[float, float | None]] = []
        self.error_hist: list[float] = []
        self.reg = LinearRegression()
        self._pf = ParticleFilter(N=128, dim_x=1) if ParticleFilter is not None else None

    @staticmethod
    def _delta(a0: float, a1: float) -> float:
        return ((a1 - a0 + 180) % 360) - 180

    def observe(self, timestamp: float, ball_angle: float, rotor_angle: float | None) -> None:
        self.ball_hist.append((timestamp, ball_angle))
        self.ball_hist = self.ball_hist[-200:]
        self.rotor_hist.append((timestamp, rotor_angle))
        self.rotor_hist = self.rotor_hist[-200:]

    def _estimate_omega(self, hist: list[tuple[float, float]]) -> float | None:
        if len(hist) < 2:
            return None
        (t0, a0), (t1, a1) = hist[-2], hist[-1]
        return self._delta(a0, a1) / max(1e-4, (t1 - t0))

    def _small_tse_rhs(self, y, _t, omega_w):
        theta_b, omega_b = y
        dom = -self.mu * self.g * np.sign(omega_b) - self.beta * (omega_b**2) - self.gamma * (omega_b - omega_w)
        return [omega_b, dom]

    def _integrate(self, theta0: float, omega0: float, omega_w: float) -> tuple[float, float]:
        t = np.linspace(0.0, 6.0, 600)
        if odeint is None:
            theta, omega = theta0, omega0
            for _ in range(600):
                dom = -self.mu * self.g * np.sign(omega) - self.beta * (omega**2) - self.gamma * (omega - omega_w)
                omega += dom * 0.01
                theta += omega * 0.01
            return theta % 360.0, omega
        sol = odeint(self._small_tse_rhs, [theta0, omega0], t, args=(omega_w,))
        omega_series = sol[:, 1]
        hit = np.where(np.abs(omega_series) < 9.0)[0]
        idx = int(hit[0]) if len(hit) else -1
        return float(sol[idx, 0] % 360.0), float(sol[idx, 1])

    def _detect_tilt_bias(self, probs: np.ndarray) -> float:
        sector = probs.reshape(37)
        top = float(np.max(sector))
        mean = float(np.mean(sector))
        ratio = top / max(mean, 1e-9)
        self.factor_tilt = float(np.clip(ratio / 2.0, 1.0, 1.6))
        return self.factor_tilt

    def auto_calibrate(self) -> None:
        if len(self.ball_hist) < 10:
            return
        X = []
        y = []
        for i in range(1, len(self.ball_hist) - 1):
            t0, a0 = self.ball_hist[i - 1]
            t1, a1 = self.ball_hist[i]
            t2, a2 = self.ball_hist[i + 1]
            w1 = self._delta(a0, a1) / max(1e-4, t1 - t0)
            w2 = self._delta(a1, a2) / max(1e-4, t2 - t1)
            alpha = (w2 - w1) / max(1e-4, t2 - t0)
            X.append([abs(w1), np.sign(w1)])
            y.append(-alpha)
        if len(X) >= 5:
            self.reg.fit(np.asarray(X), np.asarray(y))
            self.beta = float(np.clip(self.reg.coef_[0] * 0.0001, 0.0001, 0.02))
            self.mu = float(np.clip(abs(self.reg.coef_[1]) * 0.001, 0.005, 0.08))

    def _distribution_37(self, impact_angle: float) -> np.ndarray:
        wheel = self.cylinder.wheel
        center_idx = int((impact_angle / 360.0) * 37) % 37
        sigma = max(4.0, self.dispersion_std_deg)
        p = np.zeros(37, dtype=float)
        for i in range(37):
            dist = min(abs(i - center_idx), 37 - abs(i - center_idx))
            p[i] = math.exp(-(dist**2) / (2 * (sigma / 9.5) ** 2))
        p = p / p.sum()

        # ruido estocástico en deflectores/pockets
        noise = np.random.normal(0.0, 0.004, size=37)
        p = np.clip(p + noise, 1e-9, None)
        p /= p.sum()
        return p

    def predict_distribution_37(self, bankroll: float) -> dict:
        if len(self.ball_hist) < 2:
            return {
                "distribution": np.ones(37) / 37,
                "confidence": 0.0,
                "edge": 0.0,
                "top_numbers": [0],
                "tilt_factor": 1.0,
                "should_bet": False,
                "bet_amount": 0.0,
                "expected_profit_1h": 0.0,
            }

        theta0 = self.ball_hist[-1][1]
        omega_b = self._estimate_omega(self.ball_hist) or 0.0
        rotor_only = [(t, a) for t, a in self.rotor_hist if a is not None]
        omega_w = self._estimate_omega(rotor_only) if rotor_only else 0.0
        omega_w = 0.0 if omega_w is None else omega_w

        impact, _ = self._integrate(theta0, omega_b, omega_w)
        probs = self._distribution_37(impact)
        p_hit = float(np.max(probs))
        tilt_factor = self._detect_tilt_bias(probs)
        confidence = float(np.clip(1 - (np.std(probs) * 37) / 360.0 * tilt_factor, 0.0, 0.99))
        payout_neto = 35.0
        edge = (p_hit - 1 / 37.0) * payout_neto

        variance = max(0.05, float(np.var(probs) * 37.0))
        kelly = bankroll * (edge / variance) * self.bankroll_fraction
        bet = float(np.clip(kelly, 0.0, bankroll * 0.2))
        should = edge > self.edge_threshold and confidence > self.confidence_threshold

        idx_sorted = np.argsort(probs)[::-1]
        top_numbers = [self.cylinder.wheel[int(i)] for i in idx_sorted[:5]]

        expected_profit_1h = 22 * bet * edge
        return {
            "distribution": probs,
            "confidence": confidence,
            "edge": float(edge),
            "top_numbers": top_numbers,
            "tilt_factor": tilt_factor,
            "should_bet": should,
            "bet_amount": bet if should else 0.0,
            "expected_profit_1h": expected_profit_1h,
        }

    def update_hyperparams_from_config(self, cfg: dict) -> None:
        self.mu = float(cfg.get("mu", self.mu))
        self.beta = float(cfg.get("beta", self.beta))
        self.gamma = float(cfg.get("gamma", self.gamma))
        self.dispersion_std_deg = float(cfg.get("dispersion_std_deg", self.dispersion_std_deg))

    def export_calibration_state(self) -> dict:
        return {
            "mu": self.mu,
            "beta": self.beta,
            "gamma": self.gamma,
            "dispersion_std_deg": self.dispersion_std_deg,
            "updated_at": int(time.time()),
        }


class RoulettePhysicsEngine(AdvancedPhysicsEngine):
    def fit_friction(self, angle_history: list[tuple[float, float]]) -> None:
        if not angle_history:
            return
        self.ball_hist = angle_history[-100:]
        self.auto_calibrate()

    def learn_dispersion(self, errors_deg: list[float]) -> None:
        if not errors_deg:
            return
        self.dispersion_std_deg = float(np.clip(np.std(errors_deg), 5.0, 24.0))

    def predict_drop(self, now_angle: float, now_omega: float, rotor_angle: float | None, rotor_omega: float | None):
        impact, _ = self._integrate(now_angle, now_omega, rotor_omega or 0.0)
        return impact, 1.0

    def sector_from_angle(self, angle: float, span_numbers: int = 10) -> list[int | str]:
        idx = int((angle / 360.0) * len(self.cylinder.wheel))
        h = span_numbers // 2
        return [self.cylinder.wheel[(idx + i) % len(self.cylinder.wheel)] for i in range(-h, h + 1)]

    def confidence_and_span(self) -> tuple[float, int]:
        c = float(np.clip(1.0 - self.dispersion_std_deg / 40.0, 0.45, 0.95))
        span = int(np.clip(round(12 - c * 6), 8, 12))
        return c, span

    def suggest_bet(self, bankroll: float, sector: list[int | str], confidence: float) -> BetSuggestion:
        edge = max(0.0, confidence - 0.5)
        variance = max(0.2, 1.0 - confidence)
        amount = float(np.clip(bankroll * (edge / variance) * 0.5, 0.0, bankroll * 0.1))
        should = confidence > 0.68 and edge > 0.12
        msg = f"EXPERIMENTAL: {'apostar' if should else 'esperar'} sector {sector[:5]}"
        return BetSuggestion(should, confidence, sector, msg, "sector", amount)
