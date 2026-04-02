from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

import json
from pathlib import Path

import pandas as pd

try:
    import telebot
except Exception:  # pragma: no cover
    telebot = None  # type: ignore

LOG_PATH = Path("audit_log.csv")


def _load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["timestamp", "ball_angle", "rotor_angle", "confidence", "edge", "tilt_factor", "top_numbers", "bet"])


def build_bot(token: str):
    if telebot is None:
        raise RuntimeError("telebot no disponible")
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def start(m):
        bot.reply_to(m, "Auditami activo. EXPERIMENTAL. Solo investigación.")

    @bot.message_handler(commands=["report"])
    def report(m):
        df = _load_log()
        if df.empty:
            bot.reply_to(m, "Sin datos.")
            return
        tail = df.tail(10)
        roi = (tail.get("edge", pd.Series([0.0])).mean()) * 100.0
        msg = f"Últimos 10 spins: conf={tail['confidence'].mean():.3f}, edge={tail['edge'].mean():.3f}, ROI proxy={roi:.2f}%"
        bot.reply_to(m, msg)

    @bot.message_handler(commands=["bias"])
    def bias(m):
        df = _load_log()
        if df.empty:
            bot.reply_to(m, "Sin datos para bias.")
            return
        v = float(df["tilt_factor"].tail(20).mean()) if "tilt_factor" in df.columns else 1.0
        bot.reply_to(m, f"Tilt factor reciente: {v:.3f}")

    @bot.message_handler(commands=["log"])
    def export_log(m):
        df = _load_log()
        out = Path("auditami_export.csv")
        df.to_csv(out, index=False)
        with out.open("rb") as fh:
            bot.send_document(m.chat.id, fh)

    @bot.message_handler(func=lambda _: True)
    def fallback(m):
        bot.reply_to(m, "Comandos: /report /bias /log")

    return bot


def main() -> int:
    cfg = Path("config.json")
    if not cfg.exists():
        print("Falta config.json")
        return 1
    token = json.loads(cfg.read_text(encoding="utf-8")).get("telegram_token", "")
    if not token:
        print("Configurar telegram_token en config.json")
        return 1
    bot = build_bot(token)
    print("Auditami Bot iniciado")
    bot.infinity_polling(timeout=30, long_polling_timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
