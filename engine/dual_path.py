from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from engine.vision import VisionState


@dataclass
class SectorPrediction:
    mode: str
    sector_index: int
    predicted_angle: float
    confidence: float
    latency_ms: float


class ExecutionLoadBalancer:
    """Gestiona pesos de ejecución dinámicos para demo de intensidad de análisis."""

    VALID_WEIGHTS = (3, 5, 10, 50, 100)

    def __init__(self, weight: int = 10):
        self.weight = 10
        self.set_weight(weight)

    def set_weight(self, weight: int) -> None:
        self.weight = int(weight) if int(weight) in self.VALID_WEIGHTS else 10

    def stride(self) -> int:
        return {3: 1, 5: 1, 10: 2, 50: 3, 100: 5}[self.weight]

    def analysis_iterations(self) -> int:
        return {3: 1, 5: 2, 10: 4, 50: 8, 100: 12}[self.weight]

    def should_process(self, frame_idx: int) -> bool:
        return frame_idx % self.stride() == 0


class FastPixelTracker:
    """Ruta reactiva de baja latencia: tracking de punto brillante + cinemática angular."""

    def __init__(self, sector_count: int = 8, sector_span_deg: float = 45.0, friction: float = 0.975):
        self.sector_count = max(1, int(sector_count))
        self.sector_span_deg = float(max(5.0, sector_span_deg))
        self.friction = float(np.clip(friction, 0.90, 0.999))
        self._prev_angle: float | None = None
        self._prev_omega: float = 0.0
        self._prev_ts: float | None = None

    @staticmethod
    def _angle(center: tuple[int, int], point: tuple[int, int]) -> float:
        dy = float(point[1] - center[1])
        dx = float(point[0] - center[0])
        return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

    @staticmethod
    def _delta(a0: float, a1: float) -> float:
        return ((a1 - a0 + 180.0) % 360.0) - 180.0

    def _detect_brightest(self, frame: np.ndarray, center: tuple[int, int], radius: int) -> tuple[int, int] | None:
        if frame is None or frame.size == 0:
            return None
        # luminancia simple + máscara anular para priorizar borde de rueda
        lum = frame.astype(np.float32).mean(axis=2) if frame.ndim == 3 else frame.astype(np.float32)
        yy, xx = np.indices(lum.shape)
        dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
        inner = max(0.0, radius * 0.68)
        outer = max(inner + 2.0, radius * 1.08)
        mask = (dist >= inner) & (dist <= outer)
        if not np.any(mask):
            return None
        masked = np.where(mask, lum, -1.0)
        idx = int(np.argmax(masked))
        y, x = np.unravel_index(idx, masked.shape)
        return int(x), int(y)

    def process(self, state: VisionState, iterations: int = 1) -> SectorPrediction | None:
        started = time.perf_counter()
        if state.frame is None or state.wheel_center is None or state.wheel_radius is None:
            return None

        bright = self._detect_brightest(state.frame, state.wheel_center, state.wheel_radius)
        if bright is None:
            return None

        angle = self._angle(state.wheel_center, bright)
        ts = float(state.timestamp)
        dt = 0.016 if self._prev_ts is None else max(1e-3, ts - self._prev_ts)

        omega = 0.0
        alpha = 0.0
        if self._prev_angle is not None:
            omega = self._delta(self._prev_angle, angle) / dt
            alpha = (omega - self._prev_omega) / dt

        # filtro por intensidad (más peso => más pasos de refinamiento)
        for _ in range(max(1, int(iterations))):
            omega *= self.friction
            alpha *= self.friction

        t_drop = float(np.clip(abs(omega) / max(abs(alpha), 5.0) * 0.15, 0.08, 1.1))
        predicted = (angle + omega * t_drop + 0.5 * alpha * t_drop * t_drop) % 360.0
        base_span = 360.0 / self.sector_count
        span = self.sector_span_deg if self.sector_span_deg > 0 else base_span
        sector = int((predicted % 360.0) // min(span, base_span)) % self.sector_count
        conf = float(np.clip(0.25 + abs(omega) / 220.0, 0.1, 0.95))

        self._prev_angle = angle
        self._prev_omega = omega
        self._prev_ts = ts

        latency_ms = (time.perf_counter() - started) * 1000.0
        return SectorPrediction(
            mode="reactive",
            sector_index=sector,
            predicted_angle=predicted,
            confidence=conf,
            latency_ms=latency_ms,
        )


def number_to_sector(number: int, sector_count: int = 8) -> int:
    if sector_count <= 0:
        return 0
    return int((int(number) % 37) * sector_count / 37) % sector_count
