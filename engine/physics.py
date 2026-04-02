from __future__ import annotations

import time
from dataclasses import dataclass

from utils.config import PhysicsConfig


@dataclass
class Prediction:
    ball_pred: float
    rotor_pred: float | None
    t_drop: float
    ball_omega: float
    ball_alpha: float
    impact_angle: float
    confidence: float
    experimental: bool = True


class AlexBotPhysics:
    """Estimador cinemático/físico simplificado de segunda capa.

    Modelo:
      dω/dt = -k_lin*ω - k_coulomb*sign(ω)
    donde k_lin y k_coulomb se calibran con una regresión local sobre muestras recientes.
    """

    def __init__(self, cfg: PhysicsConfig | None = None):
        self.cfg = cfg or PhysicsConfig()
        self.ball_history: list[tuple[float, float]] = []
        self.rotor_history: list[tuple[float, float]] = []

    def update(self, ball_angle: float | None, rotor_angle: float | None = None):
        now = time.time()
        if ball_angle is not None:
            self.ball_history.append((now, ball_angle))
        if rotor_angle is not None:
            self.rotor_history.append((now, rotor_angle))
        self._trim()

    def _trim(self):
        max_history = self.cfg.max_history
        if len(self.ball_history) > max_history:
            self.ball_history = self.ball_history[-max_history:]
        if len(self.rotor_history) > max_history:
            self.rotor_history = self.rotor_history[-max_history:]

    @staticmethod
    def _angle_delta(a1: float, a2: float) -> float:
        return ((a2 - a1 + 180.0) % 360.0) - 180.0

    @classmethod
    def _instant_omegas(cls, history: list[tuple[float, float]]):
        omegas: list[float] = []
        times: list[float] = []
        for i in range(1, len(history)):
            t0, a0 = history[i - 1]
            t1, a1 = history[i]
            dt = t1 - t0
            if dt <= 1e-5:
                continue
            dtheta = cls._angle_delta(a0, a1)
            omegas.append(dtheta / dt)
            times.append((t0 + t1) * 0.5)
        return times, omegas

    @staticmethod
    def _lin_regression(xs: list[float], ys: list[float]) -> tuple[float, float] | tuple[None, None]:
        n = len(xs)
        if n < 2:
            return None, None
        mx = sum(xs) / n
        my = sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom <= 1e-9:
            return None, None
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        intercept = my - slope * mx
        return slope, intercept

    def _fit_ball_dynamics(self):
        if len(self.ball_history) < 6:
            return None

        times, omegas = self._instant_omegas(self.ball_history)
        if len(omegas) < 5:
            return None

        # Filtrado robusto por ventana reciente
        omegas = omegas[-self.cfg.fit_window :]
        times = times[-self.cfg.fit_window :]

        # Normalización de tiempo
        t0 = times[0]
        tx = [t - t0 for t in times]

        # Ajuste local omega(t) ≈ a*t + b
        slope, intercept = self._lin_regression(tx, omegas)
        if slope is None or intercept is None:
            return None

        omega_now = omegas[-1]
        alpha_now = slope

        # Derivamos k_lin aproximado: dω/dt = -k_lin*ω - k_coulomb*sign(ω)
        # Usamos alpha y omega actuales; k_coulomb se fija en cfg
        sign = 1.0 if omega_now >= 0 else -1.0
        k_coulomb = self.cfg.coulomb_drag
        k_lin = max(self.cfg.min_linear_drag, (-alpha_now - k_coulomb * sign) / max(abs(omega_now), 1e-6))

        return {
            'omega_now': omega_now,
            'alpha_now': alpha_now,
            'k_lin': k_lin,
            'k_coulomb': k_coulomb,
        }

    def _project_angle(self, angle_now: float, omega_now: float, t: float, k_lin: float, k_coulomb: float) -> float:
        # Integración numérica estable (Euler semi-implícito)
        dt = self.cfg.integrator_dt
        angle = angle_now
        omega = omega_now
        remaining = max(0.0, t)

        while remaining > 1e-9:
            h = min(dt, remaining)
            sign = 1.0 if omega >= 0 else -1.0
            alpha = -k_lin * omega - k_coulomb * sign
            omega = omega + alpha * h
            angle = angle + omega * h
            remaining -= h

        return angle % 360.0

    def _solve_drop_time(self, omega_now: float, k_lin: float, k_coulomb: float) -> float | None:
        # Simulación hasta umbral de caída
        dt = self.cfg.integrator_dt
        threshold = self.cfg.drop_omega_threshold
        w = omega_now
        t = 0.0
        target_sign = 1.0 if w >= 0 else -1.0

        while t <= self.cfg.max_drop_time:
            if abs(w) <= threshold:
                return t
            sign = 1.0 if w >= 0 else -1.0
            alpha = -k_lin * w - k_coulomb * sign
            w = w + alpha * dt

            # Evitar cruzar a signo contrario por sobrepaso numérico
            if sign != target_sign:
                return None
            t += dt

        return None

    def _predict_rotor(self, t_drop: float) -> float | None:
        if len(self.rotor_history) < 4:
            return None

        times, omegas = self._instant_omegas(self.rotor_history)
        if len(omegas) < 3:
            return None

        rotor_omega = sum(omegas[-3:]) / min(3, len(omegas))
        rotor_angle_now = self.rotor_history[-1][1]

        # Rotor más estable: decae suavemente
        return self._project_angle(
            angle_now=rotor_angle_now,
            omega_now=rotor_omega,
            t=t_drop,
            k_lin=self.cfg.rotor_linear_drag,
            k_coulomb=self.cfg.rotor_coulomb_drag,
        )

    def _apply_deflector_spread(self, impact_angle: float, omega_now: float) -> float:
        # Modelo heurístico: mayor velocidad => mayor dispersión angular efectiva
        spread = min(self.cfg.max_deflector_spread_deg, abs(omega_now) * self.cfg.spread_gain)
        return (impact_angle + spread * self.cfg.spread_bias) % 360.0

    def _confidence(self, omega_now: float, alpha_now: float, t_drop: float) -> float:
        # Confianza empírica [0,1]
        w_score = max(0.0, min(1.0, (abs(omega_now) - self.cfg.drop_omega_threshold) / 20.0))
        a_score = max(0.0, min(1.0, abs(alpha_now) / 25.0))
        t_score = max(0.0, min(1.0, 1.0 - (t_drop / self.cfg.max_drop_time)))
        return round(0.45 * w_score + 0.30 * a_score + 0.25 * t_score, 3)

    def get_prediction(self) -> Prediction | None:
        fit = self._fit_ball_dynamics()
        if fit is None:
            return None

        omega_now = fit['omega_now']
        alpha_now = fit['alpha_now']
        k_lin = fit['k_lin']
        k_coulomb = fit['k_coulomb']

        if abs(omega_now) < self.cfg.drop_omega_threshold:
            return None

        t_drop = self._solve_drop_time(omega_now, k_lin=k_lin, k_coulomb=k_coulomb)
        if t_drop is None or t_drop < self.cfg.min_drop_time:
            return None

        ball_now = self.ball_history[-1][1]
        impact_angle = self._project_angle(ball_now, omega_now, t_drop, k_lin=k_lin, k_coulomb=k_coulomb)
        ball_pred = self._apply_deflector_spread(impact_angle, omega_now)

        rotor_pred = self._predict_rotor(t_drop)
        confidence = self._confidence(omega_now, alpha_now, t_drop)

        return Prediction(
            ball_pred=ball_pred,
            rotor_pred=rotor_pred,
            t_drop=t_drop,
            ball_omega=omega_now,
            ball_alpha=alpha_now,
            impact_angle=impact_angle,
            confidence=confidence,
            experimental=True,
        )
