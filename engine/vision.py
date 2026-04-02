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
    phase: str = "unknown"
    det_confidence: float = 0.0
    track_stability: float = 0.0
    angular_kappa: float = 0.0


class _AngleKalman:
    def __init__(self):
        self.x = np.zeros(2, dtype=float)
        self.P = np.diag([360.0 ** 2, 200.0 ** 2])
        self.Q = np.diag([0.5, 5.0])
        self.R = np.array([[9.0]])
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
        self.predict(dt)
        H = np.array([[1.0, 0.0]])
        innov = angle_deg - float(H @ self.x)
        innov = ((innov + 180.0) % 360.0) - 180.0
        S = float((H @ self.P @ H.T + self.R)[0, 0])
        K = (self.P @ H.T) / S
        self.x = self.x + K.flatten() * innov
        self.x[0] %= 360.0
        self.P = (np.eye(2) - np.outer(K.flatten(), H)) @ self.P
        return float(self.x[0])


class _AngularEKF:
    """MEJORA GOD: EKF no lineal con estado [θ, ω, α, friction]."""

    def __init__(self):
        self.x = np.array([0.0, 0.0, 0.0, 0.015], dtype=float)
        self.P = np.diag([180.0, 120.0, 60.0, 0.02])
        self.R = np.array([[9.0]])
        self.base_q = np.array([0.25, 2.0, 1.0, 0.0008])
        self._initialized = False

    def _phase_noise_gain(self, phase: str) -> float:
        if phase == "high_speed":
            return 1.7
        if phase == "decelerating":
            return 1.0
        if phase == "dropping":
            return 2.2
        return 1.2

    def predict(self, dt: float, phase: str = "unknown") -> float | None:
        if not self._initialized:
            return None
        dt = max(1e-4, dt)
        th, om, acc, fr = self.x
        fr = float(np.clip(fr, 0.001, 0.12))
        # dinámica no lineal
        acc_new = acc - fr * om * dt
        om_new = om + acc_new * dt
        th_new = (th + om * dt + 0.5 * acc_new * dt * dt) % 360.0
        self.x = np.array([th_new, om_new, acc_new, fr])

        F = np.array(
            [
                [1.0, dt, 0.5 * dt * dt, -0.5 * om * dt * dt],
                [0.0, 1.0 - fr * dt, dt, -om * dt],
                [0.0, -fr * dt, 1.0, -om * dt],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        q_gain = self._phase_noise_gain(phase)
        Q = np.diag(self.base_q * q_gain)
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0])

    def update(self, angle_deg: float, dt: float, phase: str = "unknown") -> float:
        if not self._initialized:
            self.x[:] = [angle_deg, 0.0, 0.0, 0.015]
            self._initialized = True
            return angle_deg
        self.predict(dt, phase)
        H = np.array([[1.0, 0.0, 0.0, 0.0]])
        innov = angle_deg - float(H @ self.x)
        innov = ((innov + 180.0) % 360.0) - 180.0
        S = float((H @ self.P @ H.T + self.R)[0, 0])
        K = (self.P @ H.T) / S
        self.x = self.x + K.flatten() * innov
        self.x[0] %= 360.0
        self.P = (np.eye(4) - K @ H) @ self.P
        return float(self.x[0])


class RouletteVision:
    def __init__(
        self,
        source: str = "0",
        model_path: str = "yolov11n.pt",
        wheel_detect_interval: int = 10,
        stable_ms: int = 500,
        min_stable_samples: int = 3,
        god_mode: bool = False,
        use_ekf: bool = False,
        hybrid_detection: bool = False,
        multi_object_fallback: bool = False,
        yolo_conf_threshold: float = 0.75,
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

        # MEJORA GOD: flags totalmente opt-in
        self.god_mode = god_mode
        self.use_ekf = use_ekf or god_mode
        self.hybrid_detection = hybrid_detection or god_mode
        self.multi_object_fallback = multi_object_fallback or god_mode
        self.yolo_conf_threshold = float(np.clip(yolo_conf_threshold, 0.35, 0.95))

        self._ball_kf = _AngularEKF() if self.use_ekf else _AngleKalman()
        self._marker_kf = _AngularEKF() if self.use_ekf else _AngleKalman()

        self._prev_gray = None
        self._prev_ball = None
        self._phase = "unknown"
        self._omega_hist: list[float] = []
        self._det_conf_hist: list[float] = []

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
            self._stable: list = []
        self._stable.append((now, token))
        self._stable = self._stable[-8:]
        same = [x for x in self._stable if x[1] == token]
        if (len(same) >= self.min_stable_samples
                and (same[-1][0] - same[0][0]) >= self.stable_ms / 1000.0):
            return token
        return None

    def open(self) -> None:
        if self._use_screen:
            try:
                import mss as _mss_mod
                self._mss_instance = _mss_mod.mss()
                self._screen_running = True
                self._screen_thread = threading.Thread(target=self._screen_capture_loop, daemon=True)
                self._screen_thread.start()
            except ImportError:  # pragma: no cover
                self._use_screen = False
                cv2 = _load_cv2()
                if cv2 is not None:
                    self.cap = cv2.VideoCapture(0)
            return
        cv2 = _load_cv2()
        if cv2 is not None:
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

    def _screen_capture_loop(self) -> None:
        cv2 = _load_cv2()
        if cv2 is None or self._mss_instance is None:
            return
        monitor = self._mss_instance.monitors[1]
        while self._screen_running:
            try:
                raw = np.array(self._mss_instance.grab(monitor))
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
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
        if not self._screen_running:
            self.open()
        try:
            return self._screen_queue.get(timeout=0.1)
        except queue.Empty:
            return None, time.time()

    def _detect_wheel_hough(self, frame: np.ndarray) -> None:
        cv2 = _load_cv2()
        if cv2 is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 80,
                                   param1=120, param2=30, minRadius=40, maxRadius=420)
        if circles is not None:
            x, y, r = circles[0][0]
            self.wheel_center = (int(x), int(y))
            self.wheel_radius = int(r)

    def _homography_auto(self, frame: np.ndarray) -> None:
        cv2 = _load_cv2()
        if cv2 is None or self.wheel_center is None or self.wheel_radius is None:
            return
        # MEJORA GOD: más puntos de referencia para mayor estabilidad
        cx, cy = self.wheel_center
        r = self.wheel_radius
        points = []
        target = []
        for deg in range(0, 360, 45):
            rad = math.radians(deg)
            points.append([cx + r * math.cos(rad), cy + r * math.sin(rad)])
            target.append([256 + 230 * math.cos(rad), 256 + 230 * math.sin(rad)])
        src = np.float32(points)
        dst = np.float32(target)
        self._homography, _ = cv2.findHomography(src, dst, cv2.RANSAC)

    def _detect_optical_flow_ball(self, frame: np.ndarray) -> tuple[int, int] | None:
        cv2 = _load_cv2()
        if cv2 is None or self._prev_gray is None or self._prev_ball is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        x, y = self._prev_ball
        h, w = gray.shape
        if 0 <= y < h and 0 <= x < w:
            fx, fy = flow[y, x]
            nx = int(np.clip(x + fx, 0, w - 1))
            ny = int(np.clip(y + fy, 0, h - 1))
            return nx, ny
        return None

    def _detect_objects(self, frame: np.ndarray) -> tuple[tuple[int, int] | None, tuple[int, int] | None, float]:
        cv2 = _load_cv2()
        if cv2 is None:
            return None, None, 0.0

        ball = None
        marker = None
        det_conf = 0.0

        if self.model is not None:
            try:
                result = self.model.predict(frame, conf=0.2, verbose=False)[0]
                for i, (box, cls_idx) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    label = str(self.model.names[int(cls_idx)]).lower()
                    conf = float(result.boxes.conf[i]) if result.boxes.conf is not None else 0.0
                    if "ball" in label or "bola" in label:
                        ball = (cx, cy)
                        det_conf = max(det_conf, conf)
                    if "marker" in label or "rotor" in label:
                        marker = (cx, cy)
                        det_conf = max(det_conf, conf)
            except Exception:
                pass

        # MEJORA GOD: fallback híbrido YOLO+clásico+flow si confidence es baja
        if ball is None or (self.hybrid_detection and det_conf < self.yolo_conf_threshold):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bright = cv2.inRange(hsv, (0, 0, 220), (180, 65, 255))
            cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 8:
                    (x, y), _ = cv2.minEnclosingCircle(c)
                    ball = (int(x), int(y))
                    det_conf = max(det_conf, 0.55)

        if (ball is None) and self.hybrid_detection:
            flow_ball = self._detect_optical_flow_ball(frame)
            if flow_ball is not None:
                ball = flow_ball
                det_conf = max(det_conf, 0.5)

        # MEJORA GOD: fallback opcional track() con ByteTrack/OC-SORT
        if self.multi_object_fallback and self.model is not None and ball is None:
            try:
                tracker = "bytetrack.yaml"
                tracked = self.model.track(frame, persist=True, conf=0.15, tracker=tracker, verbose=False)[0]
                if tracked.boxes is not None and len(tracked.boxes) > 0:
                    box = tracked.boxes.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in box]
                    ball = ((x1 + x2) // 2, (y1 + y2) // 2)
                    det_conf = max(det_conf, 0.45)
            except Exception:
                pass

        return ball, marker, float(np.clip(det_conf, 0.0, 1.0))

    @staticmethod
    def _angle(center: tuple[int, int] | None, p: tuple[int, int] | None) -> float | None:
        if center is None or p is None:
            return None
        return math.degrees(math.atan2(-(p[1] - center[1]), p[0] - center[0])) % 360

    @staticmethod
    def _delta(a0: float, a1: float) -> float:
        return ((a1 - a0 + 180.0) % 360.0) - 180.0

    def _detect_phase(self, omega: float) -> str:
        abs_w = abs(omega)
        if abs_w > 180.0:
            return "high_speed"
        if abs_w > 45.0:
            return "decelerating"
        return "dropping"

    def _track_stability(self) -> float:
        if len(self._det_conf_hist) < 4:
            return 0.0
        conf_mean = float(np.mean(self._det_conf_hist[-12:]))
        conf_std = float(np.std(self._det_conf_hist[-12:]))
        stability = np.clip(conf_mean * (1.0 - conf_std), 0.0, 1.0)
        return float(stability)

    def _von_mises_kappa(self) -> float:
        if len(self._omega_hist) < 5:
            return 0.0
        sigma = max(0.1, float(np.std(self._omega_hist[-15:])))
        return float(np.clip(1.0 / (math.radians(sigma) ** 2), 0.2, 120.0))

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
        if self.wheel_center is None or (self.frame_idx % self.wheel_detect_interval == 0):
            self._detect_wheel_hough(frame)
            if self.wheel_center is not None and self._homography is None:
                self._homography_auto(frame)

        ball_raw, marker_raw, det_conf = self._detect_objects(frame)

        ball_angle_raw = self._angle(self.wheel_center, ball_raw)
        marker_angle_raw = self._angle(self.wheel_center, marker_raw)

        if ball_angle_raw is not None:
            if self.use_ekf:
                ball_angle = self._ball_kf.update(ball_angle_raw, dt, self._phase)
            else:
                ball_angle = self._ball_kf.update(ball_angle_raw, dt)
        else:
            if self.use_ekf:
                ball_angle = self._ball_kf.predict(dt, self._phase)
            else:
                ball_angle = self._ball_kf.predict(dt)

        if marker_angle_raw is not None:
            if self.use_ekf:
                marker_angle = self._marker_kf.update(marker_angle_raw, dt, self._phase)
            else:
                marker_angle = self._marker_kf.update(marker_angle_raw, dt)
        else:
            if self.use_ekf:
                marker_angle = self._marker_kf.predict(dt, self._phase)
            else:
                marker_angle = self._marker_kf.predict(dt)

        if ball_angle is not None and hasattr(self, "_last_ball_angle"):
            omega = self._delta(float(self._last_ball_angle), float(ball_angle)) / max(dt, 1e-4)
            self._omega_hist.append(omega)
            self._omega_hist = self._omega_hist[-30:]
            self._phase = self._detect_phase(omega)
        self._last_ball_angle = ball_angle

        self._det_conf_hist.append(det_conf)
        self._det_conf_hist = self._det_conf_hist[-30:]

        if ball_raw is not None:
            self._prev_ball = ball_raw
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return VisionState(
            frame=frame,
            timestamp=ts,
            wheel_center=self.wheel_center,
            wheel_radius=self.wheel_radius,
            ball_center=ball_raw,
            marker_center=marker_raw,
            ball_angle=ball_angle,
            rotor_angle=marker_angle,
            phase=self._phase,
            det_confidence=det_conf,
            track_stability=self._track_stability(),
            angular_kappa=self._von_mises_kappa(),
        )


AlexBotVision = RouletteVision
