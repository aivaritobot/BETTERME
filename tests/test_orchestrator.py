from __future__ import annotations

import time

import numpy as np

from engine.dual_path import FastPixelTracker
from engine.interaction import TargetActionManager
from engine.orchestrator import (
    RandomizedActionController,
    SourceConsistencyBayes,
    TemporalSyncAdapter,
    UnifiedInferenceOrchestrator,
)
from engine.physics import AdvancedPhysicsEngine
from engine.vision import VisionState


def test_temporal_sync_adapter_clips_dynamic_offset_to_2000ms():
    sync = TemporalSyncAdapter(max_compensation_ms=2000.0, smoothing=1.0)
    telemetry = sync.register_event(visual_ts=10.0, confirmation_ts=13.5)
    assert telemetry.drift_ms == 3500.0
    assert telemetry.dynamic_offset_ms == 2000.0
    assert sync.align_timestamp(10.0) == 12.0


def test_source_consistency_bayes_returns_15pct_target_factor():
    bayes = SourceConsistencyBayes()
    for _ in range(50):
        bayes.update(velocity_deg_s=180.0, torque_deg_s2=-80.0)
    sig = bayes.signature()
    assert sig["descent_error_factor"] == 0.85
    assert 30.0 <= sig["v0"] <= 360.0


def test_randomized_action_controller_injects_or_keeps_sector():
    ctrl = RandomizedActionController(error_injection_rate=1.0, seed=7)
    selected, mode = ctrl.select_sector(predicted_sector=3, sector_count=8, confidence=0.2)
    assert 0 <= selected <= 7
    assert mode in {"A_high_confidence", "B_error_injection_random", "B_error_injection_inverse"}


def test_unified_inference_orchestrator_light_pipeline_emits_demo_payload():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[20, 100] = [255, 255, 255]
    state = VisionState(
        frame=frame,
        timestamp=time.time(),
        wheel_center=(60, 60),
        wheel_radius=45,
        ball_center=None,
        marker_center=None,
        ball_angle=15.0,
        rotor_angle=0.0,
    )
    orchestrator = UnifiedInferenceOrchestrator(
        physics=AdvancedPhysicsEngine(),
        light_predictor=FastPixelTracker(sector_count=8, sector_span_deg=45.0),
        interaction=TargetActionManager(enabled=False),
    )
    payload = orchestrator.infer(
        state=state,
        inference_mode="reactive",
        sector_count=8,
        sector_coords=[(10, 10)] * 8,
        bankroll=100.0,
        confirmation_ts=state.timestamp + 0.35,
    )
    assert payload is not None
    assert payload["pipeline"] == "Ruta Light"
    assert "Demo Técnica de Predicción Cinemática" in str(payload["demo_label"])
    assert isinstance(payload["sync_offset_ms"], float)
