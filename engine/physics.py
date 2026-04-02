from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

import math
import time
from dataclasses import dataclass

import numpy as np

try:
    from scipy.integrate import odeint
    from scipy.stats import vonmises as _vonmises
except Exception:  # pragma: no cover
    odeint = None       # type: ignore
    _vonmises = None    # type: ignore


# ---------------------------------------------------------------------------
# Legacy dataclasses (backward compat)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Wheel maps
# ---------------------------------------------------------------------------

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
            "Voisins":   [22, 18, 29, 7, 28, 12, 35, 3, 26, 0, 32, 15, 19, 4, 21, 2, 25],
            "Orphelins": [1, 20, 14, 31, 9, 17, 34, 6],
            "Tier":      [33, 16, 24, 5, 10, 23, 8, 30, 11, 36, 13, 27],
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


# ---------------------------------------------------------------------------
# Legacy simple predictor
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PHOENIX-BESTIA V4 — Advanced Physics Engine
# ---------------------------------------------------------------------------

# Maximum possible Shannon entropy for 37 equiprobable outcomes (5.209 bits)
_MAX_ENTROPY_37 = math.log2(37)


class AdvancedPhysicsEngine:
    r"""Modelo híbrido: ODE + SMC Particle Filter para fricción + Von Mises bounce.

    Ecuación de movimiento:
        d²θ/dt² = -μg·sign(ω) - β·ω² - γ·(ω - ω_rueda)

    Calibración: Sequential Monte Carlo (256 partículas) actualiza μ y β
    en tiempo real sin regresión lineal.

    Distribución de impacto: Von Mises circular en lugar de Gaussiana,
    capturando la naturaleza periódica del bounce en deflectores.

    Confianza: Entropía de Shannon — mide cuánta "información" tiene el bot.
    """

    _N_PARTICLES = 256

    def __init__(self):
        # --- Physical constants ---
        self.mu = 0.015          # Coulomb friction coefficient
        self.beta = 0.0009       # Quadratic air drag
        self.gamma = 0.04        # Rotor coupling
        self.g = 9.81

        # --- Uncertainty & thresholds ---
        self.dispersion_std_deg = 14.0
        self.factor_tilt = 1.0
        self.edge_threshold = 0.12
        self.confidence_threshold = 0.82   # ← upgraded: Shannon entropy basis
        self.bankroll_fraction = 0.5

        # --- Wheel mapping ---
        self.cylinder = UniversalCylinderMap(mode="European")

        # --- Observation history ---
        self.ball_hist: list[tuple[float, float]] = []
        self.rotor_hist: list[tuple[float, float | None]] = []

        # --- SMC: particle swarm for friction coefficients ---
        rng = np.random.default_rng()
        self._p_mu = rng.normal(self.mu, self.mu * 0.15,
                                self._N_PARTICLES).clip(0.003, 0.1)
        self._p_beta = rng.normal(self.beta, self.beta * 0.15,
                                  self._N_PARTICLES).clip(1e-5, 0.025)
        self._p_w = np.ones(self._N_PARTICLES) / self._N_PARTICLES

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _delta(a0: float, a1: float) -> float:
        return ((a1 - a0 + 180) % 360) - 180

    def _estimate_omega(self, hist: list[tuple[float, ...]]) -> float | None:
        if len(hist) < 2:
            return None
        (t0, a0), (t1, a1) = hist[-2], hist[-1]
        if a0 is None or a1 is None:
            return None
        return self._delta(float(a0), float(a1)) / max(1e-4, t1 - t0)

    # ------------------------------------------------------------------
    # Sequential Monte Carlo — friction calibration
    # ------------------------------------------------------------------

    def _smc_update_friction(self) -> None:
        """Bayesian update of μ and β from observed deceleration."""
        if len(self.ball_hist) < 5:
            return

        # Estimate recent alpha (angular acceleration) and omega
        recent = self.ball_hist[-8:]
        alphas, omegas = [], []
        for i in range(1, len(recent) - 1):
            t0, a0 = recent[i - 1]
            t1, a1 = recent[i]
            t2, a2 = recent[i + 1]
            w1 = self._delta(a0, a1) / max(1e-4, t1 - t0)
            w2 = self._delta(a1, a2) / max(1e-4, t2 - t1)
            alpha = (w2 - w1) / max(1e-4, t2 - t0)
            alphas.append(alpha)
            omegas.append(w1)

        if len(alphas) < 2:
            return

        obs_alpha = float(np.mean(alphas))
        obs_omega = float(np.mean(omegas))

        # Predicted alpha from each particle
        pred_alpha = -(
            self._p_mu * self.g * np.sign(obs_omega)
            + self._p_beta * obs_omega ** 2
            + self.gamma * obs_omega
        )

        # Gaussian likelihood
        sigma = max(0.5, abs(obs_alpha) * 0.2)
        log_w = -0.5 * ((pred_alpha - obs_alpha) / sigma) ** 2
        log_w -= log_w.max()                     # numerical stability
        w = np.exp(log_w) + 1e-300
        w /= w.sum()

        # Effective sample size → resample if collapsed
        n_eff = 1.0 / (w ** 2).sum()
        if n_eff < self._N_PARTICLES / 2:
            rng = np.random.default_rng()
            idx = rng.choice(self._N_PARTICLES, size=self._N_PARTICLES, p=w)
            jitter_mu = self._p_mu.std() * 0.05
            jitter_beta = self._p_beta.std() * 0.05
            self._p_mu = (self._p_mu[idx]
                          + rng.normal(0, max(jitter_mu, 1e-5), self._N_PARTICLES)
                          ).clip(0.003, 0.1)
            self._p_beta = (self._p_beta[idx]
                            + rng.normal(0, max(jitter_beta, 1e-7), self._N_PARTICLES)
                            ).clip(1e-5, 0.025)
            self._p_w = np.ones(self._N_PARTICLES) / self._N_PARTICLES
        else:
            self._p_w = w

        # Update point estimates from weighted mean
        self.mu = float(np.clip(np.average(self._p_mu, weights=self._p_w), 0.005, 0.08))
        self.beta = float(np.clip(np.average(self._p_beta, weights=self._p_w), 1e-5, 0.02))

    # ------------------------------------------------------------------
    # ODE integration
    # ------------------------------------------------------------------

    def _rhs(self, y, _t, omega_w: float):
        theta_b, omega_b = y
        dom = (
            -self.mu * self.g * np.sign(omega_b)
            - self.beta * omega_b ** 2
            - self.gamma * (omega_b - omega_w)
        )
        return [omega_b, dom]

    def _integrate(self, theta0: float, omega0: float,
                   omega_w: float) -> tuple[float, float]:
        t = np.linspace(0.0, 6.0, 600)
        if odeint is None:
            theta, omega = theta0, omega0
            for _ in range(600):
                dom = (
                    -self.mu * self.g * np.sign(omega)
                    - self.beta * omega ** 2
                    - self.gamma * (omega - omega_w)
                )
                omega += dom * 0.01
                theta += omega * 0.01
            return theta % 360.0, omega
        sol = odeint(self._rhs, [theta0, omega0], t, args=(omega_w,))
        omega_s = sol[:, 1]
        hit = np.where(np.abs(omega_s) < 9.0)[0]
        idx = int(hit[0]) if len(hit) else -1
        return float(sol[idx, 0] % 360.0), float(sol[idx, 1])

    # ------------------------------------------------------------------
    # Von Mises probability distribution over 37 slots
    # ------------------------------------------------------------------

    def _distribution_37(self, impact_angle: float) -> np.ndarray:
        """Von Mises circular distribution — proper model for periodic bounce."""
        center_idx = int((impact_angle / 360.0) * 37) % 37

        # Convert dispersion (deg) → Von Mises concentration κ
        # κ → 0: uniform (max chaos); κ → ∞: spike at center
        sigma_rad = max(0.05, math.radians(self.dispersion_std_deg))
        kappa = float(np.clip(1.0 / sigma_rad ** 2, 0.3, 60.0))

        # Evaluate Von Mises PDF at 37 equally-spaced points
        slots_rad = np.linspace(0.0, 2 * math.pi, 37, endpoint=False)
        center_rad = (center_idx / 37) * 2 * math.pi

        if _vonmises is not None:
            p = _vonmises.pdf(slots_rad, kappa, loc=center_rad)
        else:
            # Fallback: wrapped Gaussian
            dists = np.array([
                min(abs(i - center_idx), 37 - abs(i - center_idx))
                for i in range(37)
            ], dtype=float)
            p = np.exp(-0.5 * (dists / (self.dispersion_std_deg / 9.5)) ** 2)

        p = np.clip(p, 1e-12, None)
        p /= p.sum()

        # Stochastic deflector scatter (small additive noise)
        noise = np.abs(np.random.normal(0.0, 0.003, 37))
        p = np.clip(p + noise, 1e-12, None)
        p /= p.sum()
        return p

    # ------------------------------------------------------------------
    # Shannon entropy — information-theoretic confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(p: np.ndarray) -> float:
        """Shannon entropy in bits."""
        return float(-np.sum(p * np.log2(np.clip(p, 1e-12, None))))

    @staticmethod
    def _entropy_confidence(p: np.ndarray) -> float:
        """Normalised confidence: 1 when fully concentrated, 0 when uniform."""
        h = AdvancedPhysicsEngine._shannon_entropy(p)
        return float(np.clip(1.0 - h / _MAX_ENTROPY_37, 0.0, 0.99))

    # ------------------------------------------------------------------
    # Tilt / bias detection
    # ------------------------------------------------------------------

    def _detect_tilt_bias(self, p: np.ndarray) -> float:
        top = float(np.max(p))
        mean = float(np.mean(p))
        self.factor_tilt = float(np.clip((top / max(mean, 1e-9)) / 2.0, 1.0, 1.6))
        return self.factor_tilt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, timestamp: float, ball_angle: float,
                rotor_angle: float | None) -> None:
        self.ball_hist.append((timestamp, ball_angle))
        self.ball_hist = self.ball_hist[-200:]
        self.rotor_hist.append((timestamp, rotor_angle))
        self.rotor_hist = self.rotor_hist[-200:]

    def auto_calibrate(self) -> None:
        """SMC-based friction calibration (replaces sklearn regression)."""
        self._smc_update_friction()

        # Also update dispersion from recent prediction variance
        if len(self.ball_hist) >= 10:
            omegas = []
            for i in range(1, min(20, len(self.ball_hist))):
                t0, a0 = self.ball_hist[i - 1]
                t1, a1 = self.ball_hist[i]
                w = self._delta(a0, a1) / max(1e-4, t1 - t0)
                omegas.append(w)
            if omegas:
                omega_std = float(np.std(omegas))
                self.dispersion_std_deg = float(
                    np.clip(omega_std * 0.4 + self.dispersion_std_deg * 0.6, 5.0, 25.0)
                )

    def predict_distribution_37(self, bankroll: float) -> dict:
        if len(self.ball_hist) < 2:
            p_uniform = np.ones(37) / 37
            return {
                "distribution": p_uniform,
                "confidence": 0.0,
                "entropy_bits": float(_MAX_ENTROPY_37),
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
        omega_w = self._estimate_omega(rotor_only) or 0.0

        impact, _ = self._integrate(theta0, omega_b, omega_w)
        probs = self._distribution_37(impact)

        # Shannon entropy confidence (information-theoretic)
        confidence = self._entropy_confidence(probs)
        entropy_bits = self._shannon_entropy(probs)

        # Tilt detection
        tilt_factor = self._detect_tilt_bias(probs)

        # Edge: expected profit per unit bet, straight-up payout 35:1
        p_hit = float(np.max(probs))
        edge = (p_hit - 1.0 / 37.0) * 35.0

        # Fractional Kelly sizing (variance from particle spread)
        variance = max(0.05, float(np.var(probs) * 37.0))
        kelly = bankroll * (edge / variance) * self.bankroll_fraction

        # Safe mode: if particle weight variance is high, cut kelly to 5%
        particle_variance = float(np.var(self._p_mu * self._p_w))
        safe_mode = particle_variance > (self.mu * 0.05) ** 2
        kelly_factor = 0.05 if safe_mode else 1.0

        bet = float(np.clip(kelly * kelly_factor, 0.0, bankroll * 0.2))

        # Bet only when Shannon confidence > 0.82 and edge > threshold
        should = (confidence > self.confidence_threshold) and (edge > self.edge_threshold)

        idx_sorted = np.argsort(probs)[::-1]
        top_numbers = [self.cylinder.wheel[int(i)] for i in idx_sorted[:5]]

        expected_profit_1h = 22 * bet * edge
        return {
            "distribution": probs,
            "confidence": confidence,
            "entropy_bits": entropy_bits,
            "edge": float(edge),
            "top_numbers": top_numbers,
            "tilt_factor": tilt_factor,
            "safe_mode": safe_mode,
            "should_bet": should,
            "bet_amount": bet if should else 0.0,
            "expected_profit_1h": expected_profit_1h,
        }

    def update_hyperparams_from_config(self, cfg: dict) -> None:
        self.mu = float(cfg.get("mu", self.mu))
        self.beta = float(cfg.get("beta", self.beta))
        self.gamma = float(cfg.get("gamma", self.gamma))
        self.dispersion_std_deg = float(cfg.get("dispersion_std_deg", self.dispersion_std_deg))
        # Re-centre SMC particles around configured priors
        rng = np.random.default_rng()
        self._p_mu = rng.normal(self.mu, self.mu * 0.15,
                                self._N_PARTICLES).clip(0.003, 0.1)
        self._p_beta = rng.normal(self.beta, self.beta * 0.15,
                                  self._N_PARTICLES).clip(1e-5, 0.025)
        self._p_w = np.ones(self._N_PARTICLES) / self._N_PARTICLES

    def export_calibration_state(self) -> dict:
        return {
            "mu": self.mu,
            "beta": self.beta,
            "gamma": self.gamma,
            "dispersion_std_deg": self.dispersion_std_deg,
            "smc_mu_std": float(np.std(self._p_mu)),
            "smc_beta_std": float(np.std(self._p_beta)),
            "updated_at": int(time.time()),
        }


# ---------------------------------------------------------------------------
# Backward-compat alias used by some tests
# ---------------------------------------------------------------------------

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

    def predict_drop(self, now_angle: float, now_omega: float,
                     rotor_angle: float | None, rotor_omega: float | None):
        impact, _ = self._integrate(now_angle, now_omega, rotor_omega or 0.0)
        return impact, 1.0

    def sector_from_angle(self, angle: float, span_numbers: int = 10) -> list[int | str]:
        idx = int((angle / 360.0) * len(self.cylinder.wheel))
        h = span_numbers // 2
        return [self.cylinder.wheel[(idx + i) % len(self.cylinder.wheel)]
                for i in range(-h, h + 1)]

    def confidence_and_span(self) -> tuple[float, int]:
        c = float(np.clip(1.0 - self.dispersion_std_deg / 40.0, 0.45, 0.95))
        span = int(np.clip(round(12 - c * 6), 8, 12))
        return c, span

    def suggest_bet(self, bankroll: float, sector: list[int | str],
                    confidence: float) -> BetSuggestion:
        edge = max(0.0, confidence - 0.5)
        variance = max(0.2, 1.0 - confidence)
        amount = float(np.clip(bankroll * (edge / variance) * 0.5, 0.0, bankroll * 0.1))
        should = confidence > 0.82 and edge > 0.12
        msg = f"EXPERIMENTAL: {'apostar' if should else 'esperar'} sector {sector[:5]}"
        return BetSuggestion(should, confidence, sector, msg, "sector", amount)
