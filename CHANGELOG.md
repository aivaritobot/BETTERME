# Changelog

## 2026-04-02

### Limpieza y unificación
- Se eliminó código legado duplicado (`core/*`, `drivers/stealth.py`).
- Se unificó arquitectura activa en `engine/`, `ui/`, `utils/`, `main.py`.

### Funcionalidad
- Se agregó soporte dual de fuente: `screen` y `video` en `engine/vision.py`.
- Se incorporó configuración explícita (`utils/config.py`) para ROI y parámetros.
- Se mejoró `main.py` con CLI, logging, control de frames y manejo de errores.
- Se mantuvo calibración en `calibrate.py` y se robusteció salida/log.
- Se dejó predicción marcada como experimental en telemetría.

### Calidad
- Se añadieron tests básicos de módulos críticos (`tests/*`).
- Se actualizó README con estado real, modos de demo y limitaciones.

### Física (upgrade)
- Se reemplazó el predictor cinemático básico por un modelo con fricción lineal + Coulomb, integración temporal y tiempo de caída.
- Se añadió predicción de impacto, dispersión heurística por deflectores y puntaje de confianza.

### Runtime/entorno
- Se desacopló la carga de OpenCV/overlay del import de `main.py`; ahora `python main.py --help` funciona sin backend GUI.
- `requirements.txt` migra a `opencv-python-headless` para evitar dependencia obligatoria de `libGL.so.1` en entornos headless.
