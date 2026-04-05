#!/bin/bash
# Doble clic en macOS. Lanza la app SIN dejar terminal visible.
cd "$(dirname "$0")"
(nohup python3 -m app.main >/dev/null 2>&1 &)
# Cerrar la ventana de Terminal que se abrió al hacer doble clic
osascript -e 'tell application "Terminal" to close (every window whose name contains "BetterMe.command")' >/dev/null 2>&1 &
exit 0
