# Architecture Notes (Desktop Refactor)

## Official entrypoint

- `app/main.py` via `AppController` is the one true desktop entrypoint.
- `launcher.py` and `BetterMe.command` delegate to this path.

## Layers

1. **Desktop shell**
   - `app/main.py`, `app/controller.py`
2. **GUI + overlay**
   - `app/gui.py`, `app/capture_overlay.py`
3. **Session orchestration**
   - `app/session.py` (`SessionController`)
4. **State/persistence**
   - `app/config_store.py` (`SettingsManager`)
5. **Error mapping**
   - `app/error_handler.py`
6. **Research engine backend**
   - `engine/*` (sin reescritura de core)
7. **Packaging**
   - `packaging/pyinstaller.desktop.spec`, `packaging/build_desktop.sh`

## UX behavior

- Startup auto-opens main window + capture area.
- Main controls include Start/Pause/Resume/Stop/Reset + unlock capture area.
- Technical output hidden under **Advanced** by default.

## Legacy path

- `main.py` queda como modo legacy/dev (no ruta principal de usuario final).
