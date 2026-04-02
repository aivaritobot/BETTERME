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

try:  # pragma: no cover
    from engine.hybrid_physics import HybridPhysicsResidual, ResidualTrainSample
except Exception:  # pragma: no cover
    HybridPhysicsResidual = None  # type: ignore
    ResidualTrainSample = None  # type: ignore


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


_MAX_ENTROPY_37 = math.log2(37)


class AdvancedPhysicsEngine:
    """MEJORA GOD: Engine híbrido físico+residual, con MC avanzado y decisión multifactor."""

    _N_PARTICLES = 256

    def __init__(self):
        self.mu = 0.015
        self.beta = 0.0009
        self.gamma = 0.04
        self.g = 9.81

        self.dispersion_std_deg = 14.0
        self.factor_tilt = 1.0
        self.edge_threshold = 0.12
        self.confidence_threshold = 0.82
        self.bankroll_fraction = 0.5

        # MEJORA GOD: parámetros opt-in
        self.god_mode = False
        self.hybrid_physics = False
        self.house_edge_adjust = 0.0
        self.payout_neto = 35.0
        self.monte_carlo_sims = 500
        self.min_entropy_signal = 0.35
        self.kelly_fraction_min = 0.25
        self.kelly_fraction_max = 0.5
        self.drawdown_guard = 0.2
        self.narrow_sector_size = 6
        self.god_single_threshold = 0.87
        self.min_max_prob = 0.175
        self.max_entropy = 2.8
        # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
        self.online_mode = False
        self.config: dict = {}

        self.cylinder = UniversalCylinderMap(mode="European")
        self.ball_hist: list[tuple[float, float]] = []
        self.rotor_hist: list[tuple[float, float | None]] = []

        self._last_det_conf = 0.0
        self._last_track_stability = 0.0
        self._last_phase = "unknown"
        self._last_kappa = 0.0
        self._calib_quality = 0.5
        self._wheel_bias_counts = np.ones(37, dtype=float)

        rng = np.random.default_rng()
        self._p_mu = rng.normal(self.mu, self.mu * 0.15, self._N_PARTICLES).clip(0.003, 0.1)
        self._p_beta = rng.normal(self.beta, self.beta * 0.15, self._N_PARTICLES).clip(1e-5, 0.025)
        self._p_w = np.ones(self._N_PARTICLES) / self._N_PARTICLES

        self._hybrid = HybridPhysicsResidual() if HybridPhysicsResidual else None
        self._residual_samples: list = []

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

    def _smc_update_friction(self) -> None:
        if len(self.ball_hist) < 5:
            return

        recent = self.ball_hist[-8:]
        # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
        if self.online_mode and self._last_phase == "dropping":
            recent = self.ball_hist[-12:]
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
        pred_alpha = -(
            self._p_mu * self.g * np.sign(obs_omega)
            + self._p_beta * obs_omega ** 2
            + self.gamma * obs_omega
        )

        sigma = max(0.5, abs(obs_alpha) * 0.2)
        log_w = -0.5 * ((pred_alpha - obs_alpha) / sigma) ** 2
        log_w -= log_w.max()
        w = np.exp(log_w) + 1e-300
        w /= w.sum()

        n_eff = 1.0 / (w ** 2).sum()
        if n_eff < self._N_PARTICLES / 2:
            rng = np.random.default_rng()
            idx = rng.choice(self._N_PARTICLES, size=self._N_PARTICLES, p=w)
            jitter_mu = self._p_mu.std() * 0.05
            jitter_beta = self._p_beta.std() * 0.05
            self._p_mu = (self._p_mu[idx] + rng.normal(0, max(jitter_mu, 1e-5), self._N_PARTICLES)).clip(0.003, 0.1)
            self._p_beta = (self._p_beta[idx] + rng.normal(0, max(jitter_beta, 1e-7), self._N_PARTICLES)).clip(1e-5, 0.025)
            self._p_w = np.ones(self._N_PARTICLES) / self._N_PARTICLES
        else:
            self._p_w = w

        self.mu = float(np.clip(np.average(self._p_mu, weights=self._p_w), 0.005, 0.08))
        self.beta = float(np.clip(np.average(self._p_beta, weights=self._p_w), 1e-5, 0.02))

    def _rhs(self, y, _t, omega_w: float):
        _theta_b, omega_b = y
        dom = -self.mu * self.g * np.sign(omega_b) - self.beta * omega_b ** 2 - self.gamma * (omega_b - omega_w)
        return [omega_b, dom]

    def _integrate(self, theta0: float, omega0: float, omega_w: float) -> tuple[float, float]:
        t = np.linspace(0.0, 6.0, 600)
        if odeint is None:
            theta, omega = theta0, omega0
            for _ in range(600):
                dom = -self.mu * self.g * np.sign(omega) - self.beta * omega ** 2 - self.gamma * (omega - omega_w)
                omega += dom * 0.01
                theta += omega * 0.01
            return theta % 360.0, omega
        sol = odeint(self._rhs, [theta0, omega0], t, args=(omega_w,))
        omega_s = sol[:, 1]
        hit = np.where(np.abs(omega_s) < 9.0)[0]
        idx = int(hit[0]) if len(hit) else -1
        return float(sol[idx, 0] % 360.0), float(sol[idx, 1])

    def _distribution_37(self, impact_angle: float) -> np.ndarray:
        center_idx = int((impact_angle / 360.0) * 37) % 37
        sigma_rad = max(0.05, math.radians(self.dispersion_std_deg))
        kappa = float(np.clip(1.0 / sigma_rad ** 2, 0.3, 60.0))
        if self._last_kappa > 0:
            kappa = float(np.clip((kappa + self._last_kappa) / 2.0, 0.3, 120.0))

        slots_rad = np.linspace(0.0, 2 * math.pi, 37, endpoint=False)
        center_rad = (center_idx / 37) * 2 * math.pi

        if _vonmises is not None:
            p = _vonmises.pdf(slots_rad, kappa, loc=center_rad)
        else:
            dists = np.array([min(abs(i - center_idx), 37 - abs(i - center_idx)) for i in range(37)], dtype=float)
            p = np.exp(-0.5 * (dists / (self.dispersion_std_deg / 9.5)) ** 2)

        p = np.clip(p, 1e-12, None)
        p /= p.sum()
        return p

    @staticmethod
    def _shannon_entropy(p: np.ndarray) -> float:
        return float(-np.sum(p * np.log2(np.clip(p, 1e-12, None))))

    # === GOD SINGLE NUMBER MODE - AÑADIDO ===
    def calculate_shannon_entropy(self, probabilities: dict[int | str, float]) -> float:
        values = np.array(list(probabilities.values()), dtype=float)
        return self._shannon_entropy(values)

    # === GOD SINGLE NUMBER MODE - AÑADIDO ===
    def angle_to_number(self, angle_rad: float) -> int | str:
        wheel = self.cylinder.wheel
        idx = int((angle_rad % (2 * math.pi)) / (2 * math.pi) * len(wheel)) % len(wheel)
        return wheel[idx]

    # === GOD SINGLE NUMBER MODE - AÑADIDO ===
    def is_dropping_phase(self) -> bool:
        return self._last_phase in {"drop", "impact", "descending", "pre_drop", "dropping"}

    # === GOD SINGLE NUMBER MODE - AÑADIDO ===
    @staticmethod
    def calculate_edge(hit_probability: float, payout_neto: float = 35.0, house_edge_adjust: float = 0.0) -> float:
        return float((hit_probability - 1.0 / 37.0) * payout_neto - house_edge_adjust)

    def _normalized_entropy(self, p: np.ndarray) -> float:
        return float(np.clip(self._shannon_entropy(p) / _MAX_ENTROPY_37, 0.0, 1.0))

    def _detect_tilt_bias(self, p: np.ndarray) -> float:
        top = float(np.max(p))
        mean = float(np.mean(p))
        self.factor_tilt = float(np.clip((top / max(mean, 1e-9)) / 2.0, 1.0, 1.6))
        return self.factor_tilt

    # === GOD SINGLE NUMBER MODE - AÑADIDO ===
    def get_god_prediction(
        self,
        particles: list[dict],
        current_angle: float,
        wheel_numbers: list[int | str],
        base_confidence: float,
    ) -> dict:
        """Devuelve sector reducido o single number según condiciones GOD."""
        if not particles:
            return {
                "sector": [],
                "top_number": None,
                "mode": "normal",
                "display_text": "Sin señal",
                "confidence": 0.0,
                "edge": 0.0,
                "max_prob": 0.0,
                "entropy": float(_MAX_ENTROPY_37),
                "color": (180, 180, 180),
            }

        prob_count = {num: 0 for num in wheel_numbers}
        for p in particles:
            landing_angle = p.get("angle", current_angle) % (2 * np.pi)
            landing_num = self.angle_to_number(landing_angle)
            if landing_num in prob_count:
                prob_count[landing_num] += 1

        total = sum(prob_count.values()) or 1
        probabilities = {num: count / total for num, count in prob_count.items()}
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        top_1 = sorted_probs[0][0]
        max_prob = float(sorted_probs[0][1])
        entropy = float(self.calculate_shannon_entropy(probabilities))

        current_edge = self.calculate_edge(max_prob, payout_neto=self.payout_neto, house_edge_adjust=self.house_edge_adjust)
        is_god_signal = (
            base_confidence > float(self.config.get("god_single_threshold", self.god_single_threshold))
            and max_prob >= float(self.config.get("min_max_prob", self.min_max_prob))
            and entropy <= float(self.config.get("max_entropy", self.max_entropy))
            and self.is_dropping_phase()
            and current_edge > 0.18
            and bool(self.config.get("god_mode", self.god_mode))
        )

        if is_god_signal:
            sector = [top_1]
            mode = "single_god"
            display_text = f"GOD → {top_1}"
            color = (0, 255, 0)
            edge_prob = max_prob
        else:
            sector_size = int(max(1, self.config.get("narrow_sector_size", self.narrow_sector_size)))
            sector = [num for num, _ in sorted_probs[:sector_size]]
            mode = "narrow_sector"
            display_text = f"Sector reducido TOP-{len(sector)}"
            color = (0, 255, 255)
            edge_prob = len(sector) / 37.0

        edge = self.calculate_edge(edge_prob, payout_neto=self.payout_neto, house_edge_adjust=self.house_edge_adjust)
        return {
            "sector": sector,
            "top_number": top_1,
            "mode": mode,
            "display_text": display_text,
            "confidence": base_confidence,
            "max_prob": max_prob,
            "entropy": entropy,
            "edge": edge,
            "color": color,
        }

    def _apply_wheel_bias(self, p: np.ndarray) -> np.ndarray:
        bias = self._wheel_bias_counts / max(1.0, np.sum(self._wheel_bias_counts))
        mixed = 0.90 * p + 0.10 * bias
        mixed = np.clip(mixed, 1e-12, None)
        mixed /= mixed.sum()
        return mixed

    def _advanced_monte_carlo(self, impact_deg: float) -> np.ndarray:
        # MEJORA GOD: MC con ruido sectorial + deflectores Von Mises
        sims = int(np.clip(self.monte_carlo_sims, 80, 2000))
        # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
        if self.online_mode and len(self.ball_hist) > 20:
            recent_std = float(np.std([a for _, a in self.ball_hist[-20:]]))
            entropy_hint = np.clip(recent_std / 90.0, 0.0, 1.0)
            sims = int(np.clip(sims * (1.1 + entropy_hint), 120, 2400))
        counts = np.zeros(37, dtype=float)
        sigma_sector = np.linspace(0.8, 1.3, 37)
        for _ in range(sims):
            slot = int((impact_deg / 360.0) * 37) % 37
            jitter = np.random.vonmises(mu=0.0, kappa=max(0.5, self._last_kappa or 4.0))
            jitter_deg = np.degrees(jitter)
            jitter_deg += np.random.normal(0.0, self.dispersion_std_deg * sigma_sector[slot])
            final_deg = (impact_deg + jitter_deg) % 360.0
            idx = int((final_deg / 360.0) * 37) % 37
            counts[idx] += 1.0
        p = counts / max(1.0, counts.sum())
        return np.clip(p, 1e-12, None)

    def _maybe_residual_correction(self, impact: float, omega_b: float, omega_w: float) -> float:
        if not (self.hybrid_physics and self._hybrid is not None):
            return impact
        features = [impact / 360.0, omega_b / 300.0, omega_w / 120.0, self.mu, self.beta * 1000.0, self.factor_tilt]
        correction = self._hybrid.predict_residual_deg(features)
        return float((impact + correction) % 360.0)

    def observe(
        self,
        timestamp: float,
        ball_angle: float,
        rotor_angle: float | None,
        det_confidence: float = 0.0,
        track_stability: float = 0.0,
        phase: str = "unknown",
        angular_kappa: float = 0.0,
    ) -> None:
        self.ball_hist.append((timestamp, ball_angle))
        self.ball_hist = self.ball_hist[-200:]
        self.rotor_hist.append((timestamp, rotor_angle))
        self.rotor_hist = self.rotor_hist[-200:]

        self._last_det_conf = float(np.clip(det_confidence, 0.0, 1.0))
        self._last_track_stability = float(np.clip(track_stability, 0.0, 1.0))
        self._last_phase = phase
        self._last_kappa = float(max(0.0, angular_kappa))

    def auto_calibrate(self) -> None:
        self._smc_update_friction()
        if len(self.ball_hist) >= 10:
            omegas = []
            for i in range(1, min(20, len(self.ball_hist))):
                t0, a0 = self.ball_hist[i - 1]
                t1, a1 = self.ball_hist[i]
                w = self._delta(a0, a1) / max(1e-4, t1 - t0)
                omegas.append(w)
            if omegas:
                omega_std = float(np.std(omegas))
                self.dispersion_std_deg = float(np.clip(omega_std * 0.4 + self.dispersion_std_deg * 0.6, 5.0, 25.0))

        mu_var = float(np.var(self._p_mu))
        beta_var = float(np.var(self._p_beta))
        self._calib_quality = float(np.clip(1.0 - (mu_var * 30.0 + beta_var * 600.0), 0.0, 1.0))

    def predict_distribution_37(self, bankroll: float) -> dict:
        if len(self.ball_hist) < 2:
            p_uniform = np.ones(37) / 37
            return {
                "distribution": p_uniform,
                "confidence": 0.0,
                "entropy_bits": float(_MAX_ENTROPY_37),
                "normalized_entropy": 1.0,
                "edge": 0.0,
                "top_numbers": [0],
                "tilt_factor": 1.0,
                "should_bet": False,
                "bet_amount": 0.0,
                "expected_profit_1h": 0.0,
                "calib_quality": self._calib_quality,
                "track_stability": self._last_track_stability,
                "det_conf": self._last_det_conf,
                "strong_signal": False,
            }

        theta0 = self.ball_hist[-1][1]
        omega_b = self._estimate_omega(self.ball_hist) or 0.0
        rotor_only = [(t, a) for t, a in self.rotor_hist if a is not None]
        omega_w = self._estimate_omega(rotor_only) or 0.0

        impact, _ = self._integrate(theta0, omega_b, omega_w)
        impact = self._maybe_residual_correction(impact, omega_b, omega_w)

        probs = self._advanced_monte_carlo(impact) if (self.god_mode or self.monte_carlo_sims > 500) else self._distribution_37(impact)
        probs = self._apply_wheel_bias(probs)

        entropy_bits = self._shannon_entropy(probs)
        normalized_entropy = self._normalized_entropy(probs)

        self._detect_tilt_bias(probs)

        p_hit = float(np.max(probs))
        edge = (p_hit - 1.0 / 37.0) * self.payout_neto - self.house_edge_adjust

        # MEJORA GOD: fórmula multifactor de confianza
        confidence = (
            0.30 * self._last_det_conf
            + 0.25 * self._last_track_stability
            + 0.25 * (1.0 - normalized_entropy)
            + 0.20 * self._calib_quality
        )
        confidence = float(np.clip(confidence, 0.0, 0.99))

        variance = max(0.05, float(np.var(probs) * 37.0))
        kelly_full = max(0.0, edge / variance)
        frac = float(np.clip(self.bankroll_fraction, self.kelly_fraction_min, self.kelly_fraction_max))

        recent_drawdown = 0.0
        if len(self.ball_hist) >= 15:
            angles = [a for _, a in self.ball_hist[-15:]]
            recent_drawdown = float(np.clip(np.std(angles) / 180.0, 0.0, 1.0))
        dd_guard = 1.0 - min(self.drawdown_guard, recent_drawdown)

        bet = float(np.clip(bankroll * kelly_full * frac * dd_guard, 0.0, bankroll * 0.2))

        strong_signal = edge > 0.15 and confidence > 0.78 and normalized_entropy < self.min_entropy_signal
        # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
        if self.online_mode:
            strong_signal = (
                confidence > 0.90
                and float(np.max(probs)) > 0.185
                and entropy_bits < 2.55
                and self.is_dropping_phase()
                and edge > 0.18
            )
        should = strong_signal if (self.god_mode or self.hybrid_physics) else (confidence > self.confidence_threshold and edge > self.edge_threshold)
        # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
        if self.online_mode and recent_drawdown > self.drawdown_guard:
            should = False
            bet = 0.0

        idx_sorted = np.argsort(probs)[::-1]
        top_numbers = [self.cylinder.wheel[int(i)] for i in idx_sorted[:5]]

        god_payload = None
        if self.god_mode:
            # === GOD SINGLE NUMBER MODE - AÑADIDO ===
            particles = [
                {"angle": (2 * math.pi * (int(i) / 37.0))}
                for i, w in enumerate(probs)
                for _ in range(int(max(1.0, float(w) * 1000)))
            ]
            god_payload = self.get_god_prediction(
                particles=particles,
                current_angle=math.radians(impact),
                wheel_numbers=self.cylinder.wheel,
                base_confidence=confidence,
            )
            if god_payload["sector"]:
                top_numbers = list(god_payload["sector"])
                edge = float(god_payload["edge"])

        best_idx = int(idx_sorted[0])
        self._wheel_bias_counts[best_idx] += 1.0

        expected_profit_1h = 22 * bet * edge
        return {
            "distribution": probs,
            "confidence": confidence,
            "entropy_bits": entropy_bits,
            "normalized_entropy": normalized_entropy,
            "edge": float(edge),
            "top_numbers": top_numbers,
            "tilt_factor": self.factor_tilt,
            "safe_mode": recent_drawdown > 0.3,
            "should_bet": should,
            "bet_amount": bet if should else 0.0,
            "expected_profit_1h": expected_profit_1h,
            "calib_quality": self._calib_quality,
            "track_stability": self._last_track_stability,
            "det_conf": self._last_det_conf,
            "strong_signal": strong_signal,
            "phase": self._last_phase,
            "mode": god_payload["mode"] if god_payload else "normal",
            "display_text": god_payload["display_text"] if god_payload else "",
            "max_prob": float(god_payload["max_prob"]) if god_payload else float(p_hit),
            "entropy": float(god_payload["entropy"]) if god_payload else float(entropy_bits),
            "color": god_payload["color"] if god_payload else (180, 180, 180),
            "top_number": god_payload["top_number"] if god_payload else (top_numbers[0] if top_numbers else None),
        }

    def update_hyperparams_from_config(self, cfg: dict) -> None:
        self.config = dict(cfg)
        self.mu = float(cfg.get("mu", self.mu))
        self.beta = float(cfg.get("beta", self.beta))
        self.gamma = float(cfg.get("gamma", self.gamma))
        self.dispersion_std_deg = float(cfg.get("dispersion_std_deg", self.dispersion_std_deg))

        self.god_mode = bool(cfg.get("god_mode", self.god_mode))
        # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
        self.online_mode = bool(cfg.get("online_mode", self.online_mode))
        self.hybrid_physics = bool(cfg.get("hybrid_physics", self.hybrid_physics or self.god_mode))
        self.house_edge_adjust = float(cfg.get("house_edge_adjust", self.house_edge_adjust))
        self.payout_neto = float(cfg.get("payout_neto", self.payout_neto))
        self.monte_carlo_sims = int(cfg.get("monte_carlo_sims", self.monte_carlo_sims))
        self.min_entropy_signal = float(cfg.get("min_entropy_signal", self.min_entropy_signal))
        self.kelly_fraction_min = float(cfg.get("kelly_fraction_min", self.kelly_fraction_min))
        self.kelly_fraction_max = float(cfg.get("kelly_fraction_max", self.kelly_fraction_max))
        self.drawdown_guard = float(cfg.get("drawdown_guard", self.drawdown_guard))
        self.narrow_sector_size = int(cfg.get("narrow_sector_size", self.narrow_sector_size))
        self.god_single_threshold = float(cfg.get("god_single_threshold", self.god_single_threshold))
        self.min_max_prob = float(cfg.get("min_max_prob", self.min_max_prob))
        self.max_entropy = float(cfg.get("max_entropy", self.max_entropy))
        # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
        if self.online_mode:
            self.narrow_sector_size = int(max(1, cfg.get("narrow_sector_size", 6)))
            self.god_single_threshold = max(self.god_single_threshold, 0.90)
            self.min_max_prob = max(self.min_max_prob, 0.185)
            self.max_entropy = min(self.max_entropy, 2.55)
            self.edge_threshold = max(self.edge_threshold, 0.18)

        rng = np.random.default_rng()
        self._p_mu = rng.normal(self.mu, self.mu * 0.15, self._N_PARTICLES).clip(0.003, 0.1)
        self._p_beta = rng.normal(self.beta, self.beta * 0.15, self._N_PARTICLES).clip(1e-5, 0.025)
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
            "calib_quality": self._calib_quality,
            "wheel_bias_counts": self._wheel_bias_counts.tolist(),
        }

    def train_hybrid_from_spins(self, spin_rows: list[dict]) -> dict:
        if not (self._hybrid and self.hybrid_physics and ResidualTrainSample):
            return {"trained": False, "reason": "hybrid physics no activo"}
        samples = []
        for row in spin_rows:
            if "pred_angle" not in row or "real_angle" not in row:
                continue
            pred = float(row["pred_angle"])
            real = float(row["real_angle"])
            residual = self._delta(pred, real)
            x = [
                pred / 360.0,
                float(row.get("omega_b", 0.0)) / 300.0,
                float(row.get("omega_w", 0.0)) / 120.0,
                self.mu,
                self.beta * 1000.0,
                self.factor_tilt,
            ]
            samples.append(ResidualTrainSample(x=x, y=residual))
        return self._hybrid.train_from_spins(samples)


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
        should = confidence > 0.82 and edge > 0.12
        msg = f"EXPERIMENTAL: {'apostar' if should else 'esperar'} sector {sector[:5]}"
        return BetSuggestion(should, confidence, sector, msg, "sector", amount)
