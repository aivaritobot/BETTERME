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

_logs_dir = Path("logs")
_logs_dir.mkdir(exist_ok=True)
_spins_dir = _logs_dir / "spins"
_spins_dir.mkdir(exist_ok=True)
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

    # MEJORA GOD: todo opt-in
    parser.add_argument("--god-mode", action="store_true", help="Activa pipeline avanzado de visión+tracking+física")
    parser.add_argument("--use-ekf", action="store_true", help="Usa EKF angular en tracking")
    parser.add_argument("--hybrid-physics", action="store_true", help="Activa residual neural physics")
    parser.add_argument("--hybrid-detection", action="store_true", help="YOLO+fallback clásico+optical flow")
    parser.add_argument("--multi-object-fallback", action="store_true", help="Fallback ByteTrack ante oclusiones")
    parser.add_argument("--yolo-conf-threshold", type=float, default=0.75)
    parser.add_argument("--mc-sims", type=int, default=500, help="N simulaciones Monte Carlo en tiempo real")
    return parser


def _load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_config(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _spin_worker(stop_event: threading.Event, args: argparse.Namespace) -> None:
    print("EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida.")

    vision = RouletteVision(
        source=args.source,
        model_path=args.yolo_model,
        god_mode=args.god_mode,
        use_ekf=args.use_ekf,
        hybrid_detection=args.hybrid_detection,
        multi_object_fallback=args.multi_object_fallback,
        yolo_conf_threshold=args.yolo_conf_threshold,
    )
    physics = AdvancedPhysicsEngine()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)

    # MEJORA GOD: flags CLI sobre-escriben config sin romper default
    cfg_runtime = dict(cfg)
    if args.god_mode:
        cfg_runtime["god_mode"] = True
        cfg_runtime["hybrid_physics"] = True
        cfg_runtime["monte_carlo_sims"] = max(args.mc_sims, 1200)
    else:
        cfg_runtime["hybrid_physics"] = bool(args.hybrid_physics or cfg_runtime.get("hybrid_physics", False))
        cfg_runtime["monte_carlo_sims"] = int(args.mc_sims or cfg_runtime.get("monte_carlo_sims", 500))

    physics.update_hyperparams_from_config(cfg_runtime)

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
            det_confidence=state.det_confidence,
            track_stability=state.track_stability,
            phase=state.phase,
            angular_kappa=state.angular_kappa,
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
                entropy=pred.get("normalized_entropy", 1.0),
                legal_warning=True,
            )

        row = {
            "timestamp": state.timestamp,
            "ball_angle": state.ball_angle,
            "rotor_angle": state.rotor_angle,
            "phase": state.phase,
            "det_conf": state.det_confidence,
            "track_stability": state.track_stability,
            "confidence": pred["confidence"],
            "entropy_bits": pred.get("entropy_bits", 0.0),
            "normalized_entropy": pred.get("normalized_entropy", 1.0),
            "edge": pred["edge"],
            "tilt_factor": pred["tilt_factor"],
            "top_numbers": pred["top_numbers"],
            "bet": pred["bet_amount"],
            "expected_profit_1h": pred["expected_profit_1h"],
            "strong_signal": pred.get("strong_signal", False),
        }
        spin_log.append(row)

        if args.voice and pred["should_bet"]:
            announce_text(f"Señal experimental. Top: {pred['top_numbers'][0]}")

        if frame_count % 5 == 0:
            result_payload = {
                "confidence": round(pred["confidence"], 3),
                "entropy_bits": round(float(pred.get("entropy_bits", 0)), 3),
                "normalized_entropy": round(float(pred.get("normalized_entropy", 1.0)), 3),
                "edge": round(float(pred["edge"]), 3),
                "top_numbers": [int(n) if isinstance(n, (int, np.integer)) else n for n in pred["top_numbers"]],
                "tilt_factor": round(float(pred["tilt_factor"]), 3),
                "safe_mode": bool(pred.get("safe_mode", False)),
                "bet_amount": round(float(pred["bet_amount"]), 2),
                "expected_profit_1h": round(float(pred["expected_profit_1h"]), 2),
                "should_bet": bool(pred["should_bet"]),
                "strong_signal": bool(pred.get("strong_signal", False)),
                "phase": state.phase,
            }
            print("RESULT:" + json.dumps(result_payload), flush=True)
            _audit_logger.info(json.dumps(result_payload, ensure_ascii=False))
            (_spins_dir / f"spin_{int(time.time()*1000)}.json").write_text(
                json.dumps({"state": row, "prediction": result_payload}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

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
