from __future__ import annotations

import numpy as np

from app.models import RuntimeSettings
from app.session import SessionController
from engine.dual_path import ExecutionLoadBalancer, FastPixelTracker, number_to_sector
from engine.vision import VisionState


def test_runtime_settings_defaults_include_dual_path_fields():
    settings = RuntimeSettings()
    assert settings.inference_mode == "analytic"
    assert settings.execution_weight == 10


def test_execution_load_balancer_supported_weights():
    lb = ExecutionLoadBalancer(50)
    assert lb.weight == 50
    assert lb.stride() == 3
    assert lb.analysis_iterations() == 8


def test_number_to_sector_is_bounded_for_8_sectors():
    assert 0 <= number_to_sector(0, 8) <= 7
    assert 0 <= number_to_sector(36, 8) <= 7


def test_resolve_sector_coordinates_uses_generated_ring_when_missing_config():
    points = SessionController._resolve_sector_coordinates({}, (100, 100), 60, 8)
    assert len(points) == 8
    assert all(isinstance(p[0], int) and isinstance(p[1], int) for p in points)


def test_fast_pixel_tracker_returns_reactive_prediction():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[20, 100] = [255, 255, 255]
    state = VisionState(
        frame=frame,
        timestamp=1.0,
        wheel_center=(60, 60),
        wheel_radius=45,
        ball_center=None,
        marker_center=None,
        ball_angle=15.0,
        rotor_angle=0.0,
    )
    tracker = FastPixelTracker(sector_count=8, sector_span_deg=45.0)
    pred = tracker.process(state)
    assert pred is not None
    assert pred.mode == "reactive"
    assert 0 <= pred.sector_index <= 7
    assert pred.latency_ms >= 0.0
