from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.models import CaptureROI, RuntimeSettings


@dataclass
class UIState:
    geometry: str = "1240x780+80+80"
    advanced_mode: bool = False
    capture_locked: bool = False


class SettingsManager:
    def __init__(self, path: Path):
        self.path = path
        self.ui = UIState()
        self.capture = CaptureROI(left=240, top=120, width=820, height=820)
        self.runtime = RuntimeSettings()
        self.last_mode = "Research / Demo / Audit"
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        ui = raw.get("ui", {})
        cap = raw.get("capture", {})
        run = raw.get("runtime", {})
        self.ui = UIState(
            geometry=str(ui.get("geometry", self.ui.geometry)),
            advanced_mode=bool(ui.get("advanced_mode", self.ui.advanced_mode)),
            capture_locked=bool(ui.get("capture_locked", self.ui.capture_locked)),
        )
        self.capture = CaptureROI(
            left=int(cap.get("left", self.capture.left)),
            top=int(cap.get("top", self.capture.top)),
            width=int(cap.get("width", self.capture.width)),
            height=int(cap.get("height", self.capture.height)),
        )
        self.runtime = RuntimeSettings(
            source=str(run.get("source", self.runtime.source)),
            bankroll=float(run.get("bankroll", self.runtime.bankroll)),
            voice=bool(run.get("voice", self.runtime.voice)),
            inference_mode=str(run.get("inference_mode", self.runtime.inference_mode)),
            execution_weight=int(run.get("execution_weight", self.runtime.execution_weight)),
        )
        self.last_mode = str(raw.get("last_mode", self.last_mode))

    def save(self) -> None:
        payload: dict[str, Any] = {
            "ui": asdict(self.ui),
            "capture": asdict(self.capture),
            "runtime": asdict(self.runtime),
            "last_mode": self.last_mode,
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
