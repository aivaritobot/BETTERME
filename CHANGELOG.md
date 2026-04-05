# Changelog

## 2026-04-05 — Desktop product-shell refactor

- Se establece `app/main.py` como entrypoint oficial para usuarios finales.
- Se implementa GUI con estados de producto (Ready, Initializing, Capturing, Paused, Error, Stopped).
- Se añade área de captura auto-abierta con lock/unlock y reset.
- Se agregan controles requeridos: Start, Pause, Resume, Stop, Reset, Reposition/Unlock, Settings, Advanced.
- Se incorpora persistencia robusta con `SettingsManager` (`app_state.json`).
- Se mejora manejo de errores con mensajes humanos y detalle técnico solo en avanzado.
- Se separan dependencias en grupos (`requirements/*.txt`).
- Se mantiene `main.py` como CLI legacy/dev y se deja fuera del flujo normal.

## Histórico

- Revisar commits previos para cambios antiguos del motor de investigación.
