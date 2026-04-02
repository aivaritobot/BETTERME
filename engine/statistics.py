from __future__ import annotations

from collections import deque


class RouletteAuditor:
    """Analizador estadístico simple por docenas para auditoría en vivo."""

    def __init__(self, window_size: int = 100):
        self.history = deque(maxlen=window_size)
        self.dozens = {
            'Docena 1': range(1, 13),
            'Docena 2': range(13, 25),
            'Docena 3': range(25, 37),
        }

    def add_number(self, n: int) -> bool:
        if 0 <= n <= 36:
            self.history.append(n)
            return True
        return False

    def get_probability_gap(self) -> dict[str, float] | str:
        if len(self.history) < 10:
            return 'Esperando más datos...'

        total = len(self.history)
        gaps: dict[str, float] = {}
        for name, dozen_range in self.dozens.items():
            seen = sum(1 for value in self.history if value in dozen_range)
            gaps[name] = seen / total

        return dict(sorted(gaps.items(), key=lambda item: item[1]))
