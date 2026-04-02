from __future__ import annotations

"""Input handler modular para capturas webcam/rtsp/screen/window (opt-in)."""

import importlib
import queue
import threading
import time
from collections import deque

import numpy as np


def _load_cv2():
    try:
        return importlib.import_module("cv2")
    except Exception:  # pragma: no cover
        return None


class InputHandler:
    # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
    def __init__(
        self,
        source: str = "0",
        capture_mode: str = "webcam",
        window_title: str = "",
        anti_jitter_size: int = 3,
        auto_crop: bool = True,
    ):
        self.source = source
        self.capture_mode = capture_mode
        self.window_title = window_title
        self.anti_jitter_size = max(1, int(anti_jitter_size))
        self.auto_crop = auto_crop

        self.cap = None
        self._mss_instance = None
        self._screen_running = False
        self._screen_queue: queue.Queue = queue.Queue(maxsize=2)
        self._screen_thread: threading.Thread | None = None
        self._recent_centers: deque[tuple[int, int]] = deque(maxlen=self.anti_jitter_size)

    # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
    def open(self) -> None:
        cv2 = _load_cv2()
        if cv2 is None:
            return

        if self.capture_mode == "screen":
            try:
                import mss as _mss_mod

                self._mss_instance = _mss_mod.mss()
                self._screen_running = True
                self._screen_thread = threading.Thread(target=self._screen_loop, daemon=True)
                self._screen_thread.start()
            except Exception:
                self.capture_mode = "webcam"

        if self.capture_mode == "window":
            # Fallback seguro: captura de escritorio completo + auto-crop.
            try:
                import mss as _mss_mod

                self._mss_instance = _mss_mod.mss()
                self._screen_running = True
                self._screen_thread = threading.Thread(target=self._screen_loop, daemon=True)
                self._screen_thread.start()
            except Exception:
                self.capture_mode = "webcam"

        if self.capture_mode in {"webcam", "rtsp"}:
            src = int(self.source) if str(self.source).isdigit() else self.source
            self.cap = cv2.VideoCapture(src)

    # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
    def close(self) -> None:
        self._screen_running = False
        if self._screen_thread is not None:
            self._screen_thread.join(timeout=1.0)
        self._screen_thread = None
        if self._mss_instance is not None:
            self._mss_instance.close()
            self._mss_instance = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
    def read(self):
        if self.capture_mode in {"screen", "window"}:
            return self._read_screen()

        cv2 = _load_cv2()
        if cv2 is None:
            return False, None, time.time()
        if self.cap is None:
            self.open()
        if self.cap is None:
            return False, None, time.time()
        ok, frame = self.cap.read()
        if not ok:
            return False, None, time.time()

        frame = self._apply_auto_crop(frame)
        return True, frame, time.time()

    def _screen_loop(self) -> None:
        cv2 = _load_cv2()
        if cv2 is None or self._mss_instance is None:
            return
        monitor = self._mss_instance.monitors[1]
        while self._screen_running:
            try:
                raw = np.array(self._mss_instance.grab(monitor))
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                frame = self._apply_auto_crop(frame)
                try:
                    self._screen_queue.put_nowait((frame, time.time()))
                except queue.Full:
                    _ = self._screen_queue.get_nowait()
                    self._screen_queue.put_nowait((frame, time.time()))
            except Exception:
                time.sleep(0.01)

    def _read_screen(self):
        if not self._screen_running:
            self.open()
        try:
            frame, ts = self._screen_queue.get(timeout=0.12)
            return True, frame, ts
        except queue.Empty:
            return False, None, time.time()

    # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
    def _apply_auto_crop(self, frame: np.ndarray) -> np.ndarray:
        if not self.auto_crop or frame is None:
            return frame
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        if self._recent_centers:
            mean_x = int(np.mean([c[0] for c in self._recent_centers]))
            mean_y = int(np.mean([c[1] for c in self._recent_centers]))
            cx, cy = mean_x, mean_y

        self._recent_centers.append((cx, cy))
        crop_size = int(min(w, h) * 0.88)
        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        return frame[y1:y2, x1:x2]
