from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AppStatus(str, Enum):
    READY = "Ready"
    INITIALIZING = "Initializing"
    CAPTURING = "Capturing"
    PAUSED = "Paused"
    ERROR = "Error"
    STOPPED = "Stopped"


@dataclass
class CaptureROI:
    left: int
    top: int
    width: int
    height: int


@dataclass
class RuntimeSettings:
    source: str = "screen"
    bankroll: float = 100.0
    voice: bool = False


@dataclass
class SessionMetrics:
    confidence: float = 0.0
    edge: float = 0.0
    entropy_bits: float = 0.0
    top_numbers: list[int] | list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.top_numbers is None:
            self.top_numbers = []
