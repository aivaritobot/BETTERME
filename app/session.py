from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Callable

from app.error_handler import to_user_error
from app.models import AppStatus, CaptureROI, RuntimeSettings
from engine.dual_path import ExecutionLoadBalancer, FastPixelTracker
from engine.interaction import TargetActionManager
from engine.physics import AdvancedPhysicsEngine
from engine.orchestrator import UnifiedInferenceOrchestrator
from engine.vision import RouletteVision
from ui.overlay import announce_text

SessionEvent = Callable[[str, dict], None]


class SessionController:
    def __init__(self, config_path: Path, emit: SessionEvent):
        self.config_path = config_path
        self.emit = emit
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._roi = CaptureROI(240, 120, 820, 820)
        self._settings = RuntimeSettings()
        self.status = AppStatus.READY

    def update_roi(self, roi: CaptureROI) -> None:
        self._roi = roi

    def update_settings(self, settings: RuntimeSettings) -> None:
        self._settings = settings

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._pause.clear()
        self.status = AppStatus.INITIALIZING
        self.emit("status", {"status": self.status.value, "message": "Inicializando sesión..."})
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        if self.status == AppStatus.CAPTURING:
            self._pause.set()
            self.status = AppStatus.PAUSED
            self.emit("status", {"status": self.status.value, "message": "Sesión en pausa."})

    def resume(self) -> None:
        if self.status == AppStatus.PAUSED:
            self._pause.clear()
            self.status = AppStatus.CAPTURING
            self.emit("status", {"status": self.status.value, "message": "Sesión reanudada."})

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        self.status = AppStatus.STOPPED
        self.emit("status", {"status": self.status.value, "message": "Sesión detenida."})

    def reset(self) -> None:
        self.stop()
        self._pause.clear()
        self.status = AppStatus.READY
        self.emit("status", {"status": self.status.value, "message": "Sistema listo para iniciar."})

    def _loop(self) -> None:
        vision: RouletteVision | None = None
        physics: AdvancedPhysicsEngine | None = None
        reactive_tracker: FastPixelTracker | None = None
        config_data: dict = {}
        try:
            config_data = self._load_engine_config()
            self.emit("health", {"ok": True, "checks": self._health_checks(config_data)})

            vision = RouletteVision(
                source=self._settings.source,
                model_path=config_data.get("yolo_model", "yolov11n"),
                god_mode=bool(config_data.get("god_mode", False)),
                use_ekf=bool(config_data.get("use_ekf", False)),
                hybrid_detection=bool(config_data.get("hybrid_detection", False)),
                multi_object_fallback=bool(config_data.get("multi_object_fallback", False)),
                yolo_conf_threshold=float(config_data.get("yolo_conf_threshold", 0.75)),
            )
            if self._settings.source in {"screen", "window"}:
                vision.set_capture_roi(self._roi.left, self._roi.top, self._roi.width, self._roi.height)

            vision.set_runtime_options(
                online_mode=bool(config_data.get("online_mode", False)),
                capture_mode=str(config_data.get("capture_mode", "screen")),
                window_title=str(config_data.get("window_title", "")),
                backend=str(config_data.get("backend", "cpu")),
                enhance_image=bool(config_data.get("enhance_image", False)),
                enhance_level=str(config_data.get("enhance_level", "medium")),
                skip_frames=int(config_data.get("skip_frames", 0)),
            )
            physics = AdvancedPhysicsEngine()
            physics.update_hyperparams_from_config(config_data)
            reactive_tracker = FastPixelTracker(
                sector_count=int(config_data.get("sector_count", 8)),
                sector_span_deg=float(config_data.get("sector_span_deg", 45.0)),
                friction=float(config_data.get("reactive_friction", 0.975)),
            )
            load_balancer = ExecutionLoadBalancer(weight=int(self._settings.execution_weight))
            action_manager = TargetActionManager(
                enabled=bool(config_data.get("enable_target_action", False)),
                jitter_px=(1, 3),
            )
            orchestrator = UnifiedInferenceOrchestrator(
                physics=physics,
                light_predictor=reactive_tracker,
                interaction=action_manager,
            )

            self.status = AppStatus.CAPTURING
            self.emit("status", {"status": self.status.value, "message": "Capturando..."})

            frame_count = 0
            while not self._stop.is_set():
                if self._pause.is_set():
                    time.sleep(0.08)
                    continue

                state = vision.read_state()
                if state is None:
                    self.emit("log", {"level": "info", "message": "No se detectan frames. Revisa fuente de captura."})
                    break
                if state.ball_angle is None:
                    continue

                frame_count += 1
                if not load_balancer.should_process(frame_count):
                    continue

                sector_count = max(1, int(config_data.get("sector_count", 8)))
                sector_coords = self._resolve_sector_coordinates(config_data, state.wheel_center, state.wheel_radius, sector_count)
                payload = orchestrator.infer(
                    state=state,
                    inference_mode=self._settings.inference_mode,
                    sector_count=sector_count,
                    sector_coords=sector_coords,
                    bankroll=self._settings.bankroll,
                )
                if payload is None:
                    continue
                payload["execution_weight"] = int(load_balancer.weight)
                payload["inference_mode"] = self._settings.inference_mode
                self.emit("metrics", payload)

                if self._settings.voice and payload["should_bet"] and payload["top_numbers"]:
                    announce_text(f"Señal experimental. Top {payload['top_numbers'][0]}")

                if frame_count % 5 == 0:
                    physics.auto_calibrate()
                    self._append_audit(payload)
        except Exception as exc:  # noqa: BLE001
            title, hint = to_user_error(exc)
            self.status = AppStatus.ERROR
            self.emit("error", {"status": self.status.value, "title": title, "hint": hint, "detail": str(exc)})
        finally:
            if vision is not None:
                vision.close()
            if physics is not None:
                config_data.update(physics.export_calibration_state())
                self._save_engine_config(config_data)
            if self.status not in {AppStatus.ERROR, AppStatus.STOPPED}:
                self.status = AppStatus.STOPPED
                self.emit("status", {"status": self.status.value, "message": "Sesión finalizada."})

    @staticmethod
    def _health_checks(config_data: dict) -> list[dict]:
        return [
            {"name": "Modo", "value": "Research / Demo / Audit", "ok": True},
            {"name": "Modelo", "value": str(config_data.get("yolo_model", "yolov11n")), "ok": True},
            {"name": "Backend", "value": str(config_data.get("backend", "cpu")), "ok": True},
        ]

    def _load_engine_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        return {}

    def _save_engine_config(self, data: dict) -> None:
        self.config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _append_audit(payload: dict) -> None:
        out = Path("logs")
        out.mkdir(exist_ok=True)
        with (out / "desktop_session.log").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), **payload}, ensure_ascii=False) + "\n")

    @staticmethod
    def _resolve_sector_coordinates(
        config_data: dict,
        wheel_center: tuple[int, int] | None,
        wheel_radius: int | None,
        sector_count: int,
    ) -> list[tuple[int, int]]:
        configured = config_data.get("target_sector_coords")
        if isinstance(configured, list) and len(configured) >= sector_count:
            points: list[tuple[int, int]] = []
            for item in configured[:sector_count]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    points.append((int(item[0]), int(item[1])))
            if len(points) == sector_count:
                return points
        if wheel_center is None or wheel_radius is None:
            return [(0, 0) for _ in range(sector_count)]
        coords: list[tuple[int, int]] = []
        r = int(wheel_radius * 0.72)
        for i in range(sector_count):
            ang = 2 * math.pi * i / sector_count
            x = int(wheel_center[0] + r * math.cos(ang))
            y = int(wheel_center[1] + r * math.sin(ang))
            coords.append((x, y))
        return coords
