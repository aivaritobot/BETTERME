from __future__ import annotations

import importlib
import re
import time
from collections import deque

import numpy as np

try:
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None  # type: ignore

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore


def _load_cv2():
    try:
        return importlib.import_module('cv2')
    except Exception:  # pragma: no cover
        return None
except Exception:  # pragma: no cover - depende de entorno gráfico
    pyautogui = None  # type: ignore

try:
    import pytesseract
except Exception:  # pragma: no cover - depende de instalación local
    pytesseract = None  # type: ignore

class RouletteVision:
    """OCR de mesa con auto-detección EU/USA y filtro temporal de estabilidad."""

    def __init__(
        self,
        region: tuple[int, int, int, int] = (0, 0, 300, 300),
        stable_ms: int = 500,
        min_stable_samples: int = 3,
    ):
        self.region = region
        self.mode = 'European'
        self.stable_ms = stable_ms
        self.min_stable_samples = min_stable_samples
        self._token_buffer: deque[tuple[float, str]] = deque(maxlen=10)
        self.last_detected: int | None = None

    def _grab_frame(self) -> np.ndarray | None:
        if pyautogui is None:  # pragma: no cover
            return None
        screenshot = pyautogui.screenshot(region=self.region)
        frame = np.array(screenshot)
        cv2 = _load_cv2()
        if cv2 is None:
            return frame[:, :, ::-1].copy()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _extract_token(raw_text: str) -> str | None:
        text = raw_text.strip()
        if '00' in text:
            return '00'
        match = re.search(r'\b([0-9]|[1-2][0-9]|3[0-6])\b', text)
        if not match:
            return None
        return match.group(1)

    @classmethod
    def _extract_number(cls, raw_text: str) -> int | None:
        token = cls._extract_token(raw_text)
        if token is None or token == '00':
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
        stable_for = now - oldest_ts
        if stable_for >= self.stable_ms / 1000:
            return token
        return None

    def scan(self) -> tuple[int | None, np.ndarray | None]:
        frame = self._grab_frame()
        cv2 = _load_cv2()
        if frame is None or pytesseract is None or cv2 is None:  # pragma: no cover
            return None, frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        text = pytesseract.image_to_string(
            thresh,
            config='--psm 6 -c tessedit_char_whitelist=0123456789',
        )

        token = self._extract_token(text)
        if token is None:
            return None, frame

        stable_token = self._promote_stable_token(token)
        if stable_token is None:
            return None, frame

        if stable_token == '00':
            self.mode = 'American'
            return None, frame

        number = int(stable_token)
        if number == self.last_detected:
            return None, frame

        self.last_detected = number
        return number, frame

    def get_last_number(self) -> int | None:
        number, _ = self.scan()
        return number

    def is_betting_open(
        self,
        sample_point: tuple[int, int] = (5, 5),
        red_threshold: int = 170,
        green_threshold: int = 150,
    ) -> bool | None:
        frame = self._grab_frame()
        if frame is None:
            return None

def _load_cv2():
    try:
        return importlib.import_module('cv2')
    except Exception:  # pragma: no cover
        return None


class RouletteVision:
    """Módulo OCR para detectar el último número en una ROI de pantalla."""

    def __init__(self, region: tuple[int, int, int, int] = (0, 0, 300, 300)):
        self.region = region
        self.last_detected: int | None = None

    def _grab_frame(self) -> np.ndarray | None:
        if pyautogui is None:  # pragma: no cover - requiere GUI real
            return None

        screenshot = pyautogui.screenshot(region=self.region)
        frame = np.array(screenshot)

        cv2 = _load_cv2()
        if cv2 is None:
            return frame[:, :, ::-1].copy()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _extract_number(raw_text: str) -> int | None:
        match = re.search(r"\d+", raw_text)
        if not match:
            return None

        value = int(match.group())
        if 0 <= value <= 36:
            return value
        return None

    def get_last_number(self) -> int | None:
        """Captura ROI, aplica OCR y devuelve un nuevo número válido 0-36."""
        frame = self._grab_frame()
        cv2 = _load_cv2()
        if frame is None or pytesseract is None or cv2 is None:  # pragma: no cover
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        text = pytesseract.image_to_string(
            thresh,
            config='--psm 6 -c tessedit_char_whitelist=0123456789',
        )

        num = self._extract_number(text.strip())
        if num is None or num == self.last_detected:
            return None

        self.last_detected = num
        return num

    def is_betting_open(
        self,
        sample_point: tuple[int, int] = (5, 5),
        red_threshold: int = 170,
        green_threshold: int = 150,
    ) -> bool | None:
        frame = self._grab_frame()
        if frame is None:
            return None

        x, y = sample_point
        h, w = frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None

        _, g, r = frame[y, x]
        b, g, r = frame[y, x]
        if r >= red_threshold and r > g:
            return False
        if g >= green_threshold and g > r:
            return True
        return None


AlexBotVision = RouletteVision
