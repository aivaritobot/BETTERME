from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

import importlib
import math
import queue
import threading
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


# ---------------------------------------------------------------------------
# VisionState
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _AngleKalman — linear 2D state [angle, omega]
# Handles YOLO dropout frames: predict-only when no detection available.
# Full EKF is overkill for 1D circular motion; a linear KF with wraparound
# innovation is numerically stable and sufficient.
# ---------------------------------------------------------------------------

class _AngleKalman:
    """Kalman filter for angular position + velocity.

    State x = [angle_deg, omega_deg_per_sec]
    Predicts position when YOLO drops frames; updates when measurement arrives.
    """

    def __init__(self):
        self.x = np.zeros(2, dtype=float)          # [angle, omega]
        self.P = np.diag([360.0 ** 2, 200.0 ** 2]) # initial covariance
        self.Q = np.diag([0.5, 5.0])               # process noise
        self.R = np.array([[9.0]])                  # measurement noise (3° std)
        self._initialized = False

    def predict(self, dt: float) -> float | None:
        if not self._initialized:
            return None
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.x[0] %= 360.0
        self.P = F @ self.P @ F.T + self.Q
        return float(self.x[0])

    def update(self, angle_deg: float, dt: float) -> float:
        if not self._initialized:
            self.x[:] = [angle_deg, 0.0]
            self._initialized = True
            return angle_deg
        # Predict step
        self.predict(dt)
        # Innovation with circular wraparound
        H = np.array([[1.0, 0.0]])
        innov = angle_deg - float(H @ self.x)
        innov = ((innov + 180.0) % 360.0) - 180.0   # wrap to [-180, 180]
        S = float((H @ self.P @ H.T + self.R)[0, 0])
        K = (self.P @ H.T) / S                        # shape (2,1)
        self.x = self.x + K.flatten() * innov
        self.x[0] %= 360.0
        self.P = (np.eye(2) - np.outer(K.flatten(), H)) @ self.P
        return float(self.x[0])


# ---------------------------------------------------------------------------
# RouletteVision
# ---------------------------------------------------------------------------

class RouletteVision:
    def __init__(
        self,
        source: str = "0",
        model_path: str = "yolov11n.pt",
        wheel_detect_interval: int = 10,
        stable_ms: int = 500,
        min_stable_samples: int = 3,
    ):
        self._use_screen = str(source).lower() == "screen"
        self._mss_instance = None
        self._screen_queue: queue.Queue = queue.Queue(maxsize=1)
        self._screen_thread: threading.Thread | None = None
        self._screen_running = False

        self.source = int(source) if str(source).isdigit() else source
        self.model = YOLO(model_path) if YOLO is not None else None
        self.cap = None
        self.wheel_center: tuple[int, int] | None = None
        self.wheel_radius: int | None = None
        self._homography = None
        self.wheel_detect_interval = wheel_detect_interval
        self.frame_idx = 0
        self.stable_ms = stable_ms
        self.min_stable_samples = min_stable_samples
        self._last_frame_time: float = time.time()

        # Angle Kalman filters (handle YOLO dropout)
        self._ball_kf = _AngleKalman()
        self._marker_kf = _AngleKalman()

    # ------------------------------------------------------------------
    # Token helpers (legacy compat)
    # ------------------------------------------------------------------

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

    def _promote_stable_token(self, token: str,
                               now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        if not hasattr(self, "_stable"):
            self._stable: list = []
        self._stable.append((now, token))
        self._stable = self._stable[-8:]
        same = [x for x in self._stable if x[1] == token]
        if (len(same) >= self.min_stable_samples
                and (same[-1][0] - same[0][0]) >= self.stable_ms / 1000.0):
            return token
        return None

    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._use_screen:
            try:
                import mss as _mss_mod
                self._mss_instance = _mss_mod.mss()
                self._screen_running = True
                self._screen_thread = threading.Thread(
                    target=self._screen_capture_loop, daemon=True
                )
                self._screen_thread.start()
            except ImportError:  # pragma: no cover
                self._use_screen = False
                cv2 = _load_cv2()
                if cv2 is not None:
                    self.cap = cv2.VideoCapture(0)
            return
        cv2 = _load_cv2()
        if cv2 is None:
            return
        self.cap = cv2.VideoCapture(self.source)

    def close(self) -> None:
        self._screen_running = False
        if self._screen_thread is not None:
            self._screen_thread.join(timeout=1.0)
        if self._mss_instance is not None:
            self._mss_instance.close()
            self._mss_instance = None
        if self.cap is not None:
            self.cap.release()

    # ------------------------------------------------------------------
    # Threaded screen capture (low-latency: capture runs independently)
    # ------------------------------------------------------------------

    def _screen_capture_loop(self) -> None:
        """Dedicated thread: captures frames and drops stale ones (maxsize=1)."""
        cv2 = _load_cv2()
        if cv2 is None or self._mss_instance is None:
            return
        monitor = self._mss_instance.monitors[1]
        while self._screen_running:
            try:
                raw = np.array(self._mss_instance.grab(monitor))
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                # Drop stale frame if consumer is slower than capture
                try:
                    self._screen_queue.put_nowait((frame, time.time()))
                except queue.Full:
                    try:
                        self._screen_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._screen_queue.put_nowait((frame, time.time()))
            except Exception:
                time.sleep(0.01)

    def _read_screen_frame(self):
        """Get latest frame from threaded capture buffer."""
        if not self._screen_running:
            self.open()
        try:
            return self._screen_queue.get(timeout=0.1)   # (frame, timestamp)
        except queue.Empty:
            return None, time.time()

    # ------------------------------------------------------------------
    # Wheel detection
    # ------------------------------------------------------------------

    def _detect_wheel_hough(self, frame: np.ndarray) -> None:
        cv2 = _load_cv2()
        if cv2 is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1.2, 80,
            param1=120, param2=30, minRadius=40, maxRadius=420,
        )
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
        src = np.float32([[cx - r, cy], [cx, cy - r],
                          [cx + r, cy], [cx, cy + r]])
        dst = np.float32([[0, 256], [256, 0], [512, 256], [256, 512]])
        self._homography, _ = cv2.findHomography(src, dst)

    # ------------------------------------------------------------------
    # Object detection (YOLO + HSV fallback)
    # ------------------------------------------------------------------

    def _detect_objects(
        self, frame: np.ndarray
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
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

        # HSV fallback for ball
        if ball is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bright = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))
            cnts, _ = cv2.findContours(
                bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                (x, y), _ = cv2.minEnclosingCircle(c)
                ball = (int(x), int(y))

        return ball, marker

    # ------------------------------------------------------------------
    # Angle helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _angle(
        center: tuple[int, int] | None, p: tuple[int, int] | None
    ) -> float | None:
        if center is None or p is None:
            return None
        return math.degrees(
            math.atan2(-(p[1] - center[1]), p[0] - center[0])
        ) % 360

    # ------------------------------------------------------------------
    # read_state — main entry point
    # ------------------------------------------------------------------

    def read_state(self) -> VisionState | None:
        cv2 = _load_cv2()
        if cv2 is None:
            return None

        now = time.time()
        dt = max(1e-4, now - self._last_frame_time)
        self._last_frame_time = now

        if self._use_screen:
            frame, ts = self._read_screen_frame()
            if frame is None:
                return None
            self.frame_idx += 1
            if self.wheel_center is None or (
                self.frame_idx % self.wheel_detect_interval == 0
            ):
                self._detect_wheel_hough(frame)
                if self.wheel_center is not None and self._homography is None:
                    self._homography_auto(frame)
        else:
            if self.cap is None:
                self.open()
            if self.cap is None:
                return None
            ok, frame = self.cap.read()
            if not ok:
                return None
            ts = now
            self.frame_idx += 1
            if self.wheel_center is None or (
                self.frame_idx % self.wheel_detect_interval == 0
            ):
                self._detect_wheel_hough(frame)
                if self.wheel_center is not None and self._homography is None:
                    self._homography_auto(frame)

        ball_raw, marker_raw = self._detect_objects(frame)

        # Raw angles from pixel positions
        ball_angle_raw = self._angle(self.wheel_center, ball_raw)
        marker_angle_raw = self._angle(self.wheel_center, marker_raw)

        # Kalman-smoothed angles (predict-only when YOLO drops the object)
        if ball_angle_raw is not None:
            ball_angle = self._ball_kf.update(ball_angle_raw, dt)
        else:
            ball_angle = self._ball_kf.predict(dt)

        if marker_angle_raw is not None:
            marker_angle = self._marker_kf.update(marker_angle_raw, dt)
        else:
            marker_angle = self._marker_kf.predict(dt)

        return VisionState(
            frame=frame,
            timestamp=ts,
            wheel_center=self.wheel_center,
            wheel_radius=self.wheel_radius,
            ball_center=ball_raw,
            marker_center=marker_raw,
            ball_angle=ball_angle,
            rotor_angle=marker_angle,
        )


AlexBotVision = RouletteVision
