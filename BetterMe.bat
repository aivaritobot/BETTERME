@echo off
REM BetterMe - Lanzador Windows (doble clic)
REM Usa pythonw.exe para no mostrar consola.
cd /d "%~dp0"
start "" pythonw.exe -m app.main
if errorlevel 1 start "" python.exe -m app.main
