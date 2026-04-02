# Problemas encontrados y resolución

## 1) Arquitectura duplicada/legado
**Problema:** coexistían módulos nuevos y legacy (`core/`, `drivers/`) causando confusión.
**Resolución:** se eliminó legado y quedó una sola ruta ejecutable.

## 2) Punto de entrada rígido
**Problema:** `main.py` no permitía modo video reproducible ni parámetros.
**Resolución:** se añadió CLI con `--source`, `--video-path`, `--roi-manual`, `--max-frames`, `--log-level`.

## 3) Falta de configuración explícita
**Problema:** constantes y parsing de ROI estaban dispersos.
**Resolución:** se creó `utils/config.py` con dataclasses de configuración y parseo validado.

## 4) Ausencia de tests
**Problema:** no había smoke tests para física/mapeo/configuración.
**Resolución:** se añadieron `tests/test_physics.py`, `tests/test_mapping.py`, `tests/test_config.py`.

## 5) Documentación poco operativa
**Problema:** README no describía todos los modos de ejecución solicitados.
**Resolución:** README actualizado con instalación, demos (video/screen), script de arranque y limitaciones reales.


## 6) Física demasiado simplificada
**Problema:** el modelo previo era casi lineal y no representaba fricción compuesta ni tiempo de caída de forma física.
**Resolución:** se incorporó dinámica `dω/dt = -k_lin*ω - k_coulomb*sign(ω)`, integración temporal, estimación de impacto, dispersión heurística por deflectores y métrica de confianza.

## 7) Fallo de libGL en arranque
**Problema:** `python main.py --help` fallaba por `libGL.so.1` al importar OpenCV GUI en import-time.
**Resolución:** imports de runtime se movieron dentro de `run()`, overlay pasó a ser opt-in (`--overlay`) y se cambió a `opencv-python-headless` para entornos sin GUI.

## 8) Hallazgos de audit no reproducibles en la base actual
**Problema:** se reportaron `IndentationError`/`SyntaxError` y tests rotos.
**Resolución:** se ejecutó auditoría automatizada (`tools/run_audit.py`) y se confirmó compilación + tests + smoke E2E en verde.