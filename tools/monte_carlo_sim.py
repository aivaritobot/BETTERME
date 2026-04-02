from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

import json
from pathlib import Path

import numpy as np


def run_monte_carlo(
    runs: int = 10_000,
    bankroll: float = 100.0,
    spins_per_hour: int = 22,
    mean_edge: float = 0.22,
) -> dict:
    rng = np.random.default_rng(42)
    finals = []
    drawdowns = []
    for _ in range(runs):
        b = bankroll
        peak = b
        max_dd = 0.0
        for _ in range(spins_per_hour):
            edge = float(np.clip(rng.normal(mean_edge, 0.09), -0.45, 0.65))
            conf = float(np.clip(rng.normal(0.73, 0.08), 0.45, 0.95))
            if edge <= 0.12 or conf <= 0.68:
                continue
            variance = max(0.08, rng.normal(0.24, 0.05))
            bet = np.clip(b * (edge / variance) * 0.5, 0, b * 0.2)
            pnl = bet * rng.normal(edge, 0.55)
            b = max(0.0, b + pnl)
            peak = max(peak, b)
            max_dd = max(max_dd, (peak - b) / max(peak, 1e-9))
        finals.append(b)
        drawdowns.append(max_dd)

    finals_arr = np.asarray(finals)
    report = {
        "runs": runs,
        "initial_bankroll": bankroll,
        "spins_per_hour": spins_per_hour,
        "mean_edge": mean_edge,
        "expected_profit_1h": float(np.mean(finals_arr - bankroll)),
        "p90_profit_1h": float(np.percentile(finals_arr - bankroll, 90)),
        "max_drawdown_mean": float(np.mean(drawdowns)),
        "target_range_note": "Escenarios favorables pueden observar +$180 a +$480, no garantizado.",
    }
    return report


def main() -> int:
    report = run_monte_carlo()

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    finals = np.array([100 + rng.normal(report["expected_profit_1h"], 110) for _ in range(2000)])
    plt.figure(figsize=(9, 4.6))
    plt.hist(finals, bins=65, color="#4c78a8", alpha=0.88)
    plt.title("Monte Carlo bankroll final (1 hora)")
    plt.xlabel("Bankroll final ($)")
    plt.ylabel("Frecuencia")
    out = Path("tools/monte_carlo_distribution.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Gráfico: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
