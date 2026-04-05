from __future__ import annotations

import random
import time
from dataclasses import dataclass


try:  # pragma: no cover
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None  # type: ignore


@dataclass
class TargetActionEvent:
    sector_index: int
    target: tuple[int, int]
    executed: bool
    reason: str


class TargetActionManager:
    """Simulación de selección de usuario con curva Bézier + jitter antropomórfico."""

    def __init__(self, enabled: bool = False, jitter_px: tuple[int, int] = (2, 3), click_delay_ms: int = 14):
        self.enabled = enabled
        self.jitter_px = jitter_px
        self.click_delay_ms = click_delay_ms

    @staticmethod
    def _bezier(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], t: float) -> tuple[int, int]:
        it = 1.0 - t
        x = (it ** 3) * p0[0] + 3 * (it ** 2) * t * p1[0] + 3 * it * (t ** 2) * p2[0] + (t ** 3) * p3[0]
        y = (it ** 3) * p0[1] + 3 * (it ** 2) * t * p1[1] + 3 * it * (t ** 2) * p2[1] + (t ** 3) * p3[1]
        return int(x), int(y)

    def _path(self, start: tuple[int, int], target: tuple[int, int], steps: int = 16) -> list[tuple[int, int]]:
        mx = (start[0] + target[0]) / 2
        my = (start[1] + target[1]) / 2
        c1 = (mx, start[1] - random.uniform(25, 90))
        c2 = (mx + random.uniform(-40, 40), my + random.uniform(-40, 40))
        return [self._bezier(start, c1, c2, target, i / max(1, steps - 1)) for i in range(steps)]

    def simulate_selection(self, sector_coords: list[tuple[int, int]], sector_index: int) -> TargetActionEvent:
        if not sector_coords:
            return TargetActionEvent(sector_index=sector_index, target=(0, 0), executed=False, reason="sin coordenadas")
        target = sector_coords[sector_index % len(sector_coords)]
        if not self.enabled:
            return TargetActionEvent(sector_index=sector_index, target=target, executed=False, reason="modo simulación deshabilitado")
        if pyautogui is None:  # pragma: no cover
            return TargetActionEvent(sector_index=sector_index, target=target, executed=False, reason="pyautogui no disponible")

        cur = pyautogui.position()
        path = self._path((int(cur.x), int(cur.y)), target)
        for x, y in path:
            jitter = random.randint(self.jitter_px[0], self.jitter_px[1])
            jx = x + random.randint(-jitter, jitter)
            jy = y + random.randint(-jitter, jitter)
            pyautogui.moveTo(jx, jy, duration=0)
            time.sleep(0.001)
        time.sleep(self.click_delay_ms / 1000.0)
        pyautogui.click()
        return TargetActionEvent(sector_index=sector_index, target=target, executed=True, reason="ok")
