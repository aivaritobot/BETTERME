from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

import importlib
import math
import time
from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


def _load_cv2():
    try:
        return importlib.import_module("cv2")
    except Exception:  # pragma: no cover
        return None


@dataclass
class VisionState:
    frame: np.ndarray | None
    timestamp: float
    wheel_center: tuple[int, int] | None
    wheel_radius: int | None
    ball_center: tuple[int, int] | None
    marker_center: tuple[int, int] | None
    ball_angle: float | None
    rotor_angle: float | None


class RouletteVision:
    def __init__(
        self,
        source: str = "0",
        model_path: str = "yolov11n.pt",
        wheel_detect_interval: int = 10,
        stable_ms: int = 500,
        min_stable_samples: int = 3,
    ):
        self.source = int(source) if str(source).isdigit() else source
        self.model = YOLO(model_path) if YOLO is not None else None
        self.cap = None
        self.wheel_center: tuple[int, int] | None = None
        self.wheel_radius: int | None = None
        self._homography = None
        self.wheel_detect_interval = wheel_detect_interval
        self.stable_ms = stable_ms
        self.min_stable_samples = min_stable_samples

    @staticmethod
    def _extract_token(raw_text: str) -> str | None:
        s = raw_text.strip()
        if "00" in s:
            return "00"
        for tok in s.split():
            if tok.isdigit() and 0 <= int(tok) <= 36:
                return tok
        return None

    @classmethod
    def _extract_number(cls, raw_text: str) -> int | None:
        token = cls._extract_token(raw_text)
        if token is None or token == "00":
            return None
        return int(token)

    def _promote_stable_token(self, token: str, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        if not hasattr(self, "_stable"):
            self._stable = []
        self._stable.append((now, token))
        self._stable = self._stable[-8:]
        same = [x for x in self._stable if x[1] == token]
        if len(same) >= self.min_stable_samples and (same[-1][0] - same[0][0]) >= self.stable_ms / 1000.0:
            return token
        return None

    def open(self) -> None:
        cv2 = _load_cv2()
        if cv2 is None:
            return
        self.cap = cv2.VideoCapture(self.source)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def _detect_wheel_hough(self, frame: np.ndarray) -> None:
        cv2 = _load_cv2()
        if cv2 is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 80, param1=120, param2=30, minRadius=40, maxRadius=420)
        if circles is not None:
            x, y, r = circles[0][0]
            self.wheel_center = (int(x), int(y))
            self.wheel_radius = int(r)

    def _homography_auto(self, frame: np.ndarray) -> None:
        cv2 = _load_cv2()
        if cv2 is None or self.wheel_center is None or self.wheel_radius is None:
            return
        cx, cy = self.wheel_center
        r = self.wheel_radius
        src = np.float32([[cx - r, cy], [cx, cy - r], [cx + r, cy], [cx, cy + r]])
        dst = np.float32([[0, 256], [256, 0], [512, 256], [256, 512]])
        self._homography, _ = cv2.findHomography(src, dst)

    def _detect_objects(self, frame: np.ndarray) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        cv2 = _load_cv2()
        if cv2 is None:
            return None, None

        ball = None
        marker = None
        if self.model is not None:
            try:
                result = self.model.predict(frame, conf=0.25, verbose=False)[0]
                for box, cls_idx in zip(result.boxes.xyxy, result.boxes.cls):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    label = str(self.model.names[int(cls_idx)]).lower()
                    if "ball" in label or "bola" in label:
                        ball = (cx, cy)
                    if "marker" in label or "rotor" in label:
                        marker = (cx, cy)
            except Exception:
                pass

        if ball is None or marker is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bright = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))
            cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts and ball is None:
                c = max(cnts, key=cv2.contourArea)
                (x, y), _ = cv2.minEnclosingCircle(c)
                ball = (int(x), int(y))

        return ball, marker

    @staticmethod
    def _angle(center: tuple[int, int] | None, p: tuple[int, int] | None) -> float | None:
        if center is None or p is None:
            return None
        return math.degrees(math.atan2(-(p[1] - center[1]), p[0] - center[0])) % 360

    def read_state(self) -> VisionState | None:
        cv2 = _load_cv2()
        if cv2 is None:
            return None
        if self.cap is None:
            self.open()
        if self.cap is None:
            return None

        ok, frame = self.cap.read()
        if not ok:
            return None

        if self.wheel_center is None:
            self._detect_wheel_hough(frame)
            self._homography_auto(frame)

        ball, marker = self._detect_objects(frame)

        return VisionState(
            frame=frame,
            timestamp=time.time(),
            wheel_center=self.wheel_center,
            wheel_radius=self.wheel_radius,
            ball_center=ball,
            marker_center=marker,
            ball_angle=self._angle(self.wheel_center, ball),
            rotor_angle=self._angle(self.wheel_center, marker),
        )


AlexBotVision = RouletteVision
