from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from engine.dual_path import FastPixelTracker, number_to_sector
from engine.interaction import TargetActionEvent, TargetActionManager
from engine.physics import AdvancedPhysicsEngine
from engine.vision import VisionState


@dataclass
class SyncTelemetry:
    drift_ms: float
    dynamic_offset_ms: float
    aligned_timestamp: float


class TemporalSyncAdapter:
    """Compensa drift temporal entre evento visual y metadata de confirmación."""

    def __init__(self, max_compensation_ms: float = 2000.0, smoothing: float = 0.2):
        self.max_compensation_ms = float(max(0.0, max_compensation_ms))
        self.smoothing = float(np.clip(smoothing, 0.01, 1.0))
        self._offset_ms = 0.0

    @property
    def offset_ms(self) -> float:
        return float(self._offset_ms)

    def register_event(self, visual_ts: float, confirmation_ts: float) -> SyncTelemetry:
        drift_ms = (float(confirmation_ts) - float(visual_ts)) * 1000.0
        clipped = float(np.clip(drift_ms, -self.max_compensation_ms, self.max_compensation_ms))
        self._offset_ms = (1.0 - self.smoothing) * self._offset_ms + self.smoothing * clipped
        aligned = float(visual_ts + self._offset_ms / 1000.0)
        return SyncTelemetry(
            drift_ms=drift_ms,
            dynamic_offset_ms=float(self._offset_ms),
            aligned_timestamp=aligned,
        )

    def align_timestamp(self, visual_ts: float) -> float:
        return float(visual_ts + self._offset_ms / 1000.0)


class SourceConsistencyBayes:
    """Estimador bayesiano simple para firma cinemática (V0 + par)."""

    def __init__(self):
        self._hist = np.ones((6, 6), dtype=float)
        self._v0_bins = np.linspace(30.0, 360.0, 7)
        self._torque_bins = np.linspace(-260.0, 260.0, 7)
        self._recent: deque[tuple[float, float]] = deque(maxlen=5)

    def update(self, velocity_deg_s: float, torque_deg_s2: float) -> None:
        v_idx = int(np.clip(np.digitize([abs(float(velocity_deg_s))], self._v0_bins)[0] - 1, 0, 5))
        t_idx = int(np.clip(np.digitize([float(torque_deg_s2)], self._torque_bins)[0] - 1, 0, 5))
        self._hist[v_idx, t_idx] += 1.0
        self._recent.append((float(velocity_deg_s), float(torque_deg_s2)))

    def signature(self) -> dict[str, float]:
        probs = self._hist / max(1.0, float(np.sum(self._hist)))
        idx = np.unravel_index(int(np.argmax(probs)), probs.shape)
        v0 = float((self._v0_bins[idx[0]] + self._v0_bins[idx[0] + 1]) * 0.5)
        torque = float((self._torque_bins[idx[1]] + self._torque_bins[idx[1] + 1]) * 0.5)
        return {
            "v0": v0,
            "torque": torque,
            "posterior": float(probs[idx]),
            "descent_error_factor": 0.85,  # target reducción de error del 15%
        }

    def correction_factor(self) -> float:
        sig = self.signature()
        posterior = float(sig["posterior"])
        # corrección conservadora 15% cuando hay firma estable
        return 1.0 - min(0.15, 0.15 * posterior * 37.0)


class RandomizedActionController:
    """Alterna entre modo A (alta confianza) y modo B (inyección de error)."""

    def __init__(self, error_injection_rate: float = 0.25, seed: int | None = None):
        self.error_injection_rate = float(np.clip(error_injection_rate, 0.0, 1.0))
        self._rng = random.Random(seed)
        self._toggle = False

    def select_sector(self, predicted_sector: int, sector_count: int, confidence: float) -> tuple[int, str]:
        sector_count = max(1, int(sector_count))
        predicted_sector = int(predicted_sector) % sector_count
        self._toggle = not self._toggle
        inject = self._toggle and (self._rng.random() < self.error_injection_rate or confidence < 0.35)
        if not inject:
            return predicted_sector, "A_high_confidence"

        if self._rng.random() < 0.5:
            rand_sector = self._rng.randrange(sector_count)
            return rand_sector, "B_error_injection_random"

        inverse = (predicted_sector + sector_count // 2) % sector_count
        return inverse, "B_error_injection_inverse"


class UnifiedInferenceOrchestrator:
    """Orquesta telemetría, inferencia heavy/light y simulación de interacción."""

    def __init__(
        self,
        physics: AdvancedPhysicsEngine,
        light_predictor: FastPixelTracker,
        interaction: TargetActionManager,
        sync_adapter: TemporalSyncAdapter | None = None,
        source_consistency: SourceConsistencyBayes | None = None,
        action_controller: RandomizedActionController | None = None,
    ):
        self.physics = physics
        self.light_predictor = light_predictor
        self.interaction = interaction
        self.sync_adapter = sync_adapter or TemporalSyncAdapter()
        self.source_consistency = source_consistency or SourceConsistencyBayes()
        self.action_controller = action_controller or RandomizedActionController()
        self._last_light_angle: float | None = None
        self._last_light_ts: float | None = None
        self._last_light_omega = 0.0

    @staticmethod
    def _delta(a0: float, a1: float) -> float:
        return ((a1 - a0 + 180.0) % 360.0) - 180.0

    def _update_kinematic_signature(self, timestamp: float, angle: float) -> dict[str, float]:
        if self._last_light_angle is None or self._last_light_ts is None:
            self._last_light_angle = float(angle)
            self._last_light_ts = float(timestamp)
            return self.source_consistency.signature()

        dt = max(1e-3, float(timestamp) - self._last_light_ts)
        omega = self._delta(self._last_light_angle, float(angle)) / dt
        torque = (omega - self._last_light_omega) / dt
        self.source_consistency.update(omega, torque)

        self._last_light_angle = float(angle)
        self._last_light_ts = float(timestamp)
        self._last_light_omega = float(omega)
        return self.source_consistency.signature()

    def infer(
        self,
        state: VisionState,
        inference_mode: str,
        sector_count: int,
        sector_coords: list[tuple[int, int]],
        bankroll: float,
        confirmation_ts: float | None = None,
    ) -> dict[str, object] | None:
        sector_count = max(1, int(sector_count))
        confirmation_ts = time.time() if confirmation_ts is None else float(confirmation_ts)

        sync = self.sync_adapter.register_event(state.timestamp, confirmation_ts)
        aligned_ts = sync.aligned_timestamp
        payload: dict[str, object] = {
            "sync_drift_ms": float(sync.drift_ms),
            "sync_offset_ms": float(sync.dynamic_offset_ms),
            "aligned_ts": float(aligned_ts),
        }

        if inference_mode == "reactive":
            light_started = time.perf_counter()
            light_pred = self.light_predictor.process(state)
            if light_pred is None:
                return None

            sig = self._update_kinematic_signature(aligned_ts, light_pred.predicted_angle)
            correction = self.source_consistency.correction_factor()
            corrected_angle = float(light_pred.predicted_angle * correction) % 360.0
            corrected_sector = int(corrected_angle / (360.0 / sector_count)) % sector_count
            latency_ms = (time.perf_counter() - light_started) * 1000.0

            selected_sector, action_mode = self.action_controller.select_sector(
                corrected_sector,
                sector_count=sector_count,
                confidence=light_pred.confidence,
            )
            action_evt: TargetActionEvent = self.interaction.simulate_selection(sector_coords, selected_sector)

            payload.update(
                {
                    "pipeline": "Ruta Light",
                    "mode": "reactive",
                    "confidence": float(light_pred.confidence),
                    "predicted_sector": int(corrected_sector),
                    "selected_sector": int(selected_sector),
                    "action_mode": action_mode,
                    "action_executed": bool(action_evt.executed),
                    "action_reason": action_evt.reason,
                    "latency_ms": float(latency_ms),
                    "target_latency_ok": bool(latency_ms < 15.0),
                    "kinematic_signature": sig,
                    "top_numbers": [int(corrected_sector)],
                    "should_bet": False,
                    "demo_label": "Demo Técnica de Predicción Cinemática con Simulación de Usuario Realista",
                }
            )
            return payload

        self.physics.observe(
            timestamp=aligned_ts,
            ball_angle=state.ball_angle,
            rotor_angle=state.rotor_angle,
            det_confidence=state.det_confidence,
            track_stability=state.track_stability,
            phase=state.phase,
            angular_kappa=state.angular_kappa,
        )
        pred = self.physics.predict_distribution_37(bankroll=bankroll)
        top_numbers = pred.get("top_numbers", [])
        top_number = int(top_numbers[0]) if top_numbers else 0
        predicted_sector = number_to_sector(top_number, sector_count=sector_count)
        selected_sector, action_mode = self.action_controller.select_sector(
            predicted_sector,
            sector_count=sector_count,
            confidence=float(pred.get("confidence", 0.0)),
        )
        action_evt = self.interaction.simulate_selection(sector_coords, selected_sector)

        payload.update(
            {
                "pipeline": "Ruta Heavy",
                "mode": "analytic",
                "confidence": float(pred.get("confidence", 0.0)),
                "entropy_bits": float(pred.get("entropy_bits", 0.0)),
                "edge": float(pred.get("edge", 0.0)),
                "top_numbers": top_numbers,
                "bet_amount": float(pred.get("bet_amount", 0.0)),
                "expected_profit_1h": float(pred.get("expected_profit_1h", 0.0)),
                "should_bet": bool(pred.get("should_bet", False)),
                "predicted_sector": int(predicted_sector),
                "selected_sector": int(selected_sector),
                "action_mode": action_mode,
                "action_executed": bool(action_evt.executed),
                "action_reason": action_evt.reason,
                "latency_ms": 0.0,
                "demo_label": "Demo Técnica de Predicción Cinemática con Simulación de Usuario Realista",
            }
        )
        return payload
