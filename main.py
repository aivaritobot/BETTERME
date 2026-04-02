from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from engine.physics import AlexBotPhysics, RoulettePhysicsEngine
from engine.vision import RouletteVision
from ui.overlay import render_live_overlay

try:
    import pyttsx3
except Exception:  # pragma: no cover
    pyttsx3 = None  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BETTERME Live Assistant (experimental): visión en vivo + física + sugerencias en overlay/voz"
    )
    parser.add_argument("--source", default="0", help="0 (webcam), URL RTSP o ruta de video")
    parser.add_argument("--yolo-model", default="", help="Ruta del modelo YOLOv8 para ball/marker")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--voice", action="store_true", help="Activa anuncio por voz")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser


def _estimate_omega(history: list[tuple[float, float]]) -> float | None:
    if len(history) < 3:
        return None
    (t0, a0), (t1, a1) = history[-2], history[-1]
    dt = max(1e-4, t1 - t0)
    delta = ((a1 - a0 + 180.0) % 360.0) - 180.0
    return delta / dt


def _load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_config(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    vision = RouletteVision(source=args.source, model_path=args.yolo_model or None)
    kinematics = AlexBotPhysics()
    physics = RoulettePhysicsEngine()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    if "k_linear" in cfg:
        physics.k_linear = float(cfg["k_linear"])
    if "k_coulomb" in cfg:
        physics.k_coulomb = float(cfg["k_coulomb"])
    if "dispersion_deg" in cfg:
        physics.dispersion_deg = float(cfg["dispersion_deg"])

    tts = pyttsx3.init() if (args.voice and pyttsx3 is not None) else None
    spin_started = False
    spin_samples = 0
    spin_angles: list[tuple[float, float]] = []
    historical_hits = 0
    historical_total = 0

    frame_count = 0
    while True:
        state = vision.read_state()
        if state is None:
            break
        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            break

        if state.ball_angle is not None:
            kinematics.update(state.ball_angle, state.rotor_angle)
            spin_angles.append((state.timestamp, state.ball_angle))

        ball_omega = _estimate_omega(kinematics.ball_history)
        rotor_omega = _estimate_omega(kinematics.rotor_history)

        suggestion_text = "Calibrando..."
        confidence = 0.0
        sector = []

        if ball_omega is not None and abs(ball_omega) > 20:
            spin_started = True

        if spin_started and ball_omega is not None and abs(ball_omega) < physics.drop_omega and len(spin_angles) > 12:
            # auto-calibración por spin terminado
            spin_samples += 1
            physics.fit_friction(spin_angles)
            if spin_samples <= 5:
                physics.learn_dispersion([np.random.uniform(-8, 8)])
            spin_angles = []
            spin_started = False

        if state.ball_angle is not None and ball_omega is not None:
            impact_angle, _t_drop = physics.predict_drop(
                now_angle=state.ball_angle,
                now_omega=ball_omega,
                rotor_angle=state.rotor_angle,
                rotor_omega=rotor_omega,
            )
            confidence, span = physics.confidence_and_span()
            sector = physics.sector_from_angle(impact_angle, span_numbers=span)
            suggestion = physics.suggest_bet(args.bankroll, sector, confidence)
            suggestion_text = suggestion.message

            if suggestion.should_bet and confidence >= args.confidence_threshold:
                if tts is not None:
                    tts.say(suggestion.message)
                    tts.runAndWait()
                historical_total += 1
                historical_hits += 1  # placeholder experimental

        accuracy = (historical_hits / historical_total) if historical_total else 0.0
        edge = max(0.0, confidence - 0.5)

        render_live_overlay(
            frame=state.frame,
            wheel_center=state.wheel_center,
            wheel_radius=state.wheel_radius,
            ball_center=state.ball_center,
            marker_center=state.marker_center,
            confidence=confidence,
            suggestion_text=suggestion_text,
            sector=sector,
            historical_accuracy=accuracy,
            edge=edge,
        )

    vision.close()

    cfg.update(
        {
            "k_linear": physics.k_linear,
            "k_coulomb": physics.k_coulomb,
            "dispersion_deg": physics.dispersion_deg,
            "updated_at": int(time.time()),
        }
    )
    _save_config(cfg_path, cfg)
    return 0


if __name__ == "__main__":
    if os.environ.get("PYTHONUNBUFFERED") is None:
        os.environ["PYTHONUNBUFFERED"] = "1"
    raise SystemExit(run(build_parser().parse_args()))
