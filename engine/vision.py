from __future__ import annotations

import importlib
import re

import numpy as np

try:
    import pyautogui
except Exception:  # pragma: no cover - depende de entorno gráfico
    pyautogui = None  # type: ignore

try:
    import pytesseract
except Exception:  # pragma: no cover - depende de instalación local
    pytesseract = None  # type: ignore


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

        b, g, r = frame[y, x]
        if r >= red_threshold and r > g:
            return False
        if g >= green_threshold and g > r:
            return True
        return None


AlexBotVision = RouletteVision
