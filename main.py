from __future__ import annotations

import argparse
import logging

from engine.bankroll import RiskManager
from engine.roulette import RouletteAuditor


logging.basicConfig(
    filename='audit_log.csv',
    level=logging.INFO,
    format='%(asctime)s,%(message)s',
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ALEXBOT AUDIT SYSTEM')
    parser.add_argument('--window-size', type=int, default=100, help='Tamaño de ventana estadística')
    parser.add_argument('--capital', type=float, default=None, help='Capital inicial para ejecutar sin prompt interactivo')
    return parser


def run_bot(window_size: int = 100, initial_capital: float | None = None) -> int:
    print('--- ALEXBOT AUDIT SYSTEM ACTIVATED ---')

    capital = initial_capital if initial_capital is not None else float(input('Capital total: '))
    engine = RouletteAuditor(window_size=window_size)
    risk = RiskManager(initial_capital=capital, stop_loss=capital * 0.3, take_profit=capital * 0.5)

    while risk.session_active:
        try:
            entry = input("\nNúmero salido (o 'q' para salir): ")
            if entry.lower() == 'q':
                break

            num = int(entry)
            if not engine.add_number(num):
                print('ERROR: número fuera de rango (0-36).')
                continue

            gaps = engine.get_probability_gap()
            print(f'Probabilidades actuales: {gaps}')

            if isinstance(gaps, dict) and gaps:
                suggestion = next(iter(gaps.keys()))
            else:
                suggestion = 'N/A'

            logging.info('Capital: %s, Sugerencia: %s', risk.capital, suggestion)

            # Ejemplo de flujo de apuesta (desactivado por defecto):
            # bet = float(input('Monto apuesta: '))
            # ok, message = risk.validate_bet(bet)
            # print(message)
            # if not ok:
            #     break
            # win = float(input('Ganancia/Pérdida de esta ronda: '))
            # risk.update_capital(win)

        except ValueError:
            print('ERROR: Introduce un número válido.')
        except KeyboardInterrupt:
            break

    return 0


def run(args: argparse.Namespace) -> int:
    return run_bot(window_size=args.window_size, initial_capital=args.capital)


if __name__ == '__main__':
    cli = build_parser().parse_args()
    raise SystemExit(run(cli))
