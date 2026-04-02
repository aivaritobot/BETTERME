from __future__ import annotations


class RiskManager:
    """Controla riesgo de sesión con reglas de stop-loss y take-profit."""

    def __init__(self, initial_capital: float, stop_loss: float, take_profit: float):
        self.capital = initial_capital
        self.stop_loss = initial_capital - stop_loss
        self.take_profit = initial_capital + take_profit
        self.session_active = True

    def validate_bet(self, bet_amount: float):
        if self.capital - bet_amount <= self.stop_loss:
            self.session_active = False
            return False, 'STOP LOSS ALCANZADO. Retira fondos.'

        if self.capital >= self.take_profit:
            self.session_active = False
            return False, 'OBJETIVO CUMPLIDO. Cierra sesión.'

        return True, 'Apuesta permitida.'

    def update_capital(self, net_gain: float):
        self.capital += net_gain
