from __future__ import annotations

from collections import deque

import numpy as np


class RouletteAuditor:
    """Analizador estadístico de resultados de ruleta."""

    def __init__(self, window_size: int = 100):
        self.history = deque(maxlen=window_size)
        self.stats = {
            'red': [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36],
            'black': [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35],
            'dozens': {1: range(1, 13), 2: range(13, 25), 3: range(25, 37)},
        }

    def add_number(self, n: int) -> bool:
        if 0 <= n <= 36:
            self.history.append(n)
            return True
        return False

    def get_probability_gap(self):
        """Calcula los sectores más desviados de su frecuencia teórica."""
        if len(self.history) < 10:
            return "Esperando más datos..."

        counts = np.bincount(list(self.history), minlength=37)
        freq = counts / len(self.history)
        _ = freq  # reservado para análisis más finos por número individual

        gaps: dict[str, float] = {}
        for dozen_id, dozen_range in self.stats['dozens'].items():
            occurrence = sum(1 for value in self.history if value in dozen_range)
            gaps[f'Docena {dozen_id}'] = occurrence / len(self.history)

        return dict(sorted(gaps.items(), key=lambda item: item[1]))
