from __future__ import annotations

from collections import deque

import numpy as np


RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_NUMBERS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}


class RouletteAuditor:
    """Analizador estadístico simple por docenas para auditoría en vivo."""

    def __init__(self, window_size: int = 100):
        self.history: deque[int] = deque(maxlen=window_size)
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

    def get_gap_signals(self) -> dict[str, dict[str, float]]:
        """Gap positivo = por debajo de frecuencia teórica (posible rebalance)."""
        total = len(self.history)
        if total < 10:
            return {}

        signals: dict[str, dict[str, float]] = {}
        for name, numbers in self.dozens.items():
            observed = sum(1 for n in self.history if n in numbers) / total
            expected = 12 / 37
            signals[name] = {'observed': observed, 'expected': expected, 'gap': expected - observed}

        red_obs = sum(1 for n in self.history if n in RED_NUMBERS) / total
        black_obs = sum(1 for n in self.history if n in BLACK_NUMBERS) / total
        signals['Rojo'] = {'observed': red_obs, 'expected': 18 / 37, 'gap': 18 / 37 - red_obs}
        signals['Negro'] = {'observed': black_obs, 'expected': 18 / 37, 'gap': 18 / 37 - black_obs}
        return signals

    def get_probability_gap(self) -> dict[str, float] | str:
        if len(self.history) < 10:
            return 'Esperando más datos...'
        signals = self.get_gap_signals()
        return {k: round(v['gap'], 4) for k, v in sorted(signals.items(), key=lambda item: item[1]['gap'], reverse=True)}

    def number_frequency(self) -> np.ndarray:
        """Frecuencia observada para cada número 0-36."""
        total = max(1, len(self.history))
        counts = np.bincount(list(self.history), minlength=37)
        return counts / total
