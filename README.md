# BETTERME Desktop (Research / Demo / Audit)

BETTERME es una aplicación de escritorio para investigación/demostración/auditoría visual.
**No es una herramienta de juego automatizado** y su uso está limitado a laboratorio/demo.

## Plataformas

- Probado en Linux (entorno de desarrollo actual).
- Soporte objetivo: macOS y Linux con Python 3.10+.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lanzamiento (usuario no técnico)

### Doble clic

- `BetterMe.command` abre la app desktop directamente.

### Flujo al iniciar

1. Se abre la ventana principal.
2. Se abre automáticamente el **área de captura**.
3. El usuario solo mueve/redimensiona esa área.
4. Pulsa **Start**.

Sin terminal y sin flags para el uso normal.

## Controles principales

- **Start**: iniciar sesión.
- **Pause**: pausar captura.
- **Resume**: reanudar sesión.
- **Stop**: detener sesión.
- **Reset**: volver a estado limpio.
- **Reposition / Unlock capture area**: desbloquear y recolocar el recuadro.
- **Settings**: ayuda rápida de ajustes.
- **Advanced**: mostrar logs técnicos (oculto por defecto).

## Estado y persistencia

La app guarda automáticamente:

- tamaño/posición de ventana principal,
- tamaño/posición del área de captura,
- bloqueo/desbloqueo del área,
- ajustes básicos (fuente, bankroll, voz),
- preferencia de panel avanzado.

Archivo: `app_state.json`.

## Solución de problemas rápida

- Si no hay video: verifica la fuente seleccionada.
- Si aparece error de librerías gráficas (`libGL.so.1`): instala dependencias OpenGL del sistema.
- Si faltan módulos Python: reinstala dependencias con `pip install -r requirements.txt`.

## Build de escritorio

```bash
./packaging/build_desktop.sh
```

Salida esperada: `dist/BETTERME`.

## Estructura del proyecto

- `app/` → shell de producto desktop (entrypoint, GUI, controladores, estado, errores).
- `engine/` → visión/física backend (lógica de investigación existente).
- `ui/` → utilidades de overlay/voz heredadas.
- `packaging/` → build PyInstaller.
- `tools/`, `dashboard.py`, `auditami_bot.py` → herramientas avanzadas/experimentales (fuera del flujo normal).

## Notas avanzadas (dev)

- CLI legacy sigue disponible en `main.py` para pruebas internas.
- El flujo oficial para usuarios finales es **solo desktop app**.
- Dependencias separadas por grupos en `requirements/`.
