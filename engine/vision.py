from __future__ import annotations

import importlib
import math
import re
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


def _load_cv2():
    try:
        return importlib.import_module("cv2")
    except Exception:  # pragma: no cover
        return None


try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


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


class _Kalman2D:
    def __init__(self):
        cv2 = _load_cv2()
        self.filter = None
        if cv2 is not None:
            kf = cv2.KalmanFilter(4, 2)
            kf.transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                np.float32,
            )
            kf.measurementMatrix = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                np.float32,
            )
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2e-1
            self.filter = kf

    def update(self, point: tuple[int, int] | None) -> tuple[int, int] | None:
        if self.filter is None:
            return point

        pred = self.filter.predict()
        out = (int(pred[0]), int(pred[1]))
        if point is not None:
            m = np.array([[np.float32(point[0])], [np.float32(point[1])]])
            corr = self.filter.correct(m)
            out = (int(corr[0]), int(corr[1]))
        return out


class RouletteVision:
    """Visión en vivo: webcam/RTSP + auto rueda (Hough) + detector YOLO opcional."""

    def __init__(
        self,
        source: str = "0",
        model_path: str | None = None,
        wheel_detect_interval: int = 10,
        stable_ms: int = 500,
        min_stable_samples: int = 3,
    ):
        self.source = self._parse_source(source)
        self.cap = None
        self.model = YOLO(model_path) if (YOLO is not None and model_path) else None
        self.mode = "European"
        self.wheel_detect_interval = wheel_detect_interval
        self.frame_idx = 0
        self.wheel_center: tuple[int, int] | None = None
        self.wheel_radius: int | None = None

        self.ball_filter = _Kalman2D()
        self.marker_filter = _Kalman2D()
        self.last_detected: int | None = None
        self._token_buffer: deque[tuple[float, str]] = deque(maxlen=10)
        self.stable_ms = stable_ms
        self.min_stable_samples = min_stable_samples

    @staticmethod
    def _parse_source(value: str):
        return int(value) if value.isdigit() else value

    @staticmethod
    def _extract_token(raw_text: str) -> str | None:
        text = raw_text.strip()
        if "00" in text:
            return "00"
        match = re.search(r"\b([0-9]|[1-2][0-9]|3[0-6])\b", text)
        return match.group(1) if match else None

    @classmethod
    def _extract_number(cls, raw_text: str) -> int | None:
        token = cls._extract_token(raw_text)
        if token is None or token == "00":
            return None
        value = int(token)
        return value if 0 <= value <= 36 else None

    def _promote_stable_token(self, token: str, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        self._token_buffer.append((now, token))

        trailing: list[tuple[float, str]] = []
        for ts, value in reversed(self._token_buffer):
            if value != token:
                break
            trailing.append((ts, value))

        if len(trailing) < self.min_stable_samples:
            return None

        oldest_ts = trailing[-1][0]
        return token if (now - oldest_ts) >= self.stable_ms / 1000 else None

    def open(self) -> None:
        cv2 = _load_cv2()
        if cv2 is None:  # pragma: no cover
            return
        self.cap = cv2.VideoCapture(self.source)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def _detect_wheel(self, frame: np.ndarray) -> None:
        cv2 = _load_cv2()
        if cv2 is None:  # pragma: no cover
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(40, min(frame.shape[:2]) // 3),
            param1=120,
            param2=30,
            minRadius=max(40, min(frame.shape[:2]) // 8),
            maxRadius=max(60, min(frame.shape[:2]) // 2),
        )
        if circles is not None:
            x, y, r = circles[0][0]
            self.wheel_center = (int(x), int(y))
            self.wheel_radius = int(r)

    def _detect_objects_fallback(self, frame: np.ndarray) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        cv2 = _load_cv2()
        if cv2 is None:
            return None, None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # bola: puntos muy brillantes
        bright = cv2.inRange(hsv, (0, 0, 220), (180, 70, 255))
        cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            (x, y), _ = cv2.minEnclosingCircle(c)
            ball = (int(x), int(y))

        # marcador rotor: verde
        green = cv2.inRange(hsv, (35, 60, 60), (90, 255, 255))
        cnts2, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker = None
        if cnts2:
            c2 = max(cnts2, key=cv2.contourArea)
            (mx, my), _ = cv2.minEnclosingCircle(c2)
            marker = (int(mx), int(my))

        return ball, marker

    def _detect_objects(self, frame: np.ndarray) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        if self.model is not None:
            try:
                result = self.model.predict(frame, verbose=False, conf=0.3)[0]
                ball, marker = None, None
                for box, cls_idx in zip(result.boxes.xyxy, result.boxes.cls):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cls_name = self.model.names[int(cls_idx)]
                    if cls_name.lower() in {"ball", "bola"}:
                        ball = (cx, cy)
                    elif cls_name.lower() in {"marker", "rotor_marker", "marcador"}:
                        marker = (cx, cy)
                if ball is not None or marker is not None:
                    return ball, marker
            except Exception:
                pass
        return self._detect_objects_fallback(frame)

    @staticmethod
    def _angle_from_center(center: tuple[int, int] | None, point: tuple[int, int] | None) -> float | None:
        if center is None or point is None:
            return None
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return math.degrees(math.atan2(-dy, dx)) % 360

    def read_state(self) -> VisionState | None:
        cv2 = _load_cv2()
        if cv2 is None:  # pragma: no cover
            return None
        if self.cap is None:
            self.open()
        if self.cap is None:
            return None

        ok, frame = self.cap.read()
        if not ok:
            return None

        self.frame_idx += 1
        if self.wheel_center is None or (self.frame_idx % self.wheel_detect_interval == 0):
            self._detect_wheel(frame)

        ball_raw, marker_raw = self._detect_objects(frame)
        ball = self.ball_filter.update(ball_raw)
        marker = self.marker_filter.update(marker_raw)

        return VisionState(
            frame=frame,
            timestamp=time.time(),
            wheel_center=self.wheel_center,
            wheel_radius=self.wheel_radius,
            ball_center=ball,
            marker_center=marker,
            ball_angle=self._angle_from_center(self.wheel_center, ball),
            rotor_angle=self._angle_from_center(self.wheel_center, marker),
        )


AlexBotVision = RouletteVision
