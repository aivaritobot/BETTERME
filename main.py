from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

import argparse
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from engine.physics import AdvancedPhysicsEngine
from engine.vision import RouletteVision
from tools.monte_carlo_sim import run_monte_carlo
from ui.overlay import announce_text, render_stealth_overlay

# Prodigy audit log — records each prediction for variance/drift analysis
_logs_dir = Path("logs")
_logs_dir.mkdir(exist_ok=True)
_audit_logger = logging.getLogger("prodigy_audit")
if not _audit_logger.handlers:
    _fh = logging.FileHandler(_logs_dir / "prodigy_audit.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _audit_logger.addHandler(_fh)
    _audit_logger.setLevel(logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BETTERME BESTIA (experimental): investigación de dinámica de ruleta con visión + física + auditoría"
    )
    parser.add_argument("--source", default="0", help="0 webcam, screen, archivo de video o RTSP")
    parser.add_argument("--yolo-model", default="yolov11n.pt", help="Modelo YOLO para tracking")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--audit-only", action="store_true", help="No muestra overlay en vivo; solo logging")
    parser.add_argument("--simulate", action="store_true", help="Ejecuta simulación Monte Carlo 10k runs")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--voice", action="store_true")
    return parser


def _load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_config(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _spin_worker(stop_event: threading.Event, args: argparse.Namespace) -> None:
    print("EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida.")
    vision = RouletteVision(source=args.source, model_path=args.yolo_model)
    physics = AdvancedPhysicsEngine()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    physics.update_hyperparams_from_config(cfg)

    frame_count = 0
    spin_log: list[dict] = []
    while not stop_event.is_set():
        state = vision.read_state()
        if state is None:
            break
        frame_count += 1
        if args.max_frames and frame_count > args.max_frames:
            break

        if state.ball_angle is None:
            continue

        physics.observe(
            timestamp=state.timestamp,
            ball_angle=state.ball_angle,
            rotor_angle=state.rotor_angle,
        )
        pred = physics.predict_distribution_37(bankroll=args.bankroll)

        if not args.audit_only:
            render_stealth_overlay(
                frame=state.frame,
                wheel_center=state.wheel_center,
                wheel_radius=state.wheel_radius,
                confidence=pred["confidence"],
                edge=pred["edge"],
                top_numbers=pred["top_numbers"],
                legal_warning=True,
            )

        row = {
            "timestamp": state.timestamp,
            "ball_angle": state.ball_angle,
            "rotor_angle": state.rotor_angle,
            "confidence": pred["confidence"],
            "edge": pred["edge"],
            "tilt_factor": pred["tilt_factor"],
            "top_numbers": pred["top_numbers"],
            "bet": pred["bet_amount"],
            "expected_profit_1h": pred["expected_profit_1h"],
        }
        spin_log.append(row)

        if args.voice and pred["should_bet"]:
            announce_text(f"Señal experimental. Top: {pred['top_numbers'][0]}")

        if frame_count % 5 == 0:
            result_payload = {
                "confidence": round(pred["confidence"], 3),
                "entropy_bits": round(float(pred.get("entropy_bits", 0)), 3),
                "edge": round(float(pred["edge"]), 3),
                "top_numbers": [int(n) if isinstance(n, (int, np.integer)) else n
                                for n in pred["top_numbers"]],
                "tilt_factor": round(float(pred["tilt_factor"]), 3),
                "safe_mode": bool(pred.get("safe_mode", False)),
                "bet_amount": round(float(pred["bet_amount"]), 2),
                "expected_profit_1h": round(float(pred["expected_profit_1h"]), 2),
                "should_bet": bool(pred["should_bet"]),
            }
            print("RESULT:" + json.dumps(result_payload), flush=True)
            _audit_logger.info(json.dumps(result_payload))

        if frame_count % 5 == 0:
            physics.auto_calibrate()

    vision.close()

    cfg.update(physics.export_calibration_state())
    _save_config(cfg_path, cfg)

    if spin_log:
        import pandas as pd

        out = Path("audit_log.csv")
        pd.DataFrame(spin_log).to_csv(out, index=False)
        print(f"Audit log exportado en {out}")


def run(args: argparse.Namespace) -> int:
    if args.simulate:
        report = run_monte_carlo(runs=10_000, bankroll=100.0, spins_per_hour=22, mean_edge=0.22)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    stop_event = threading.Event()
    worker = threading.Thread(target=_spin_worker, args=(stop_event, args), daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
        worker.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
