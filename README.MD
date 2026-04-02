# ALEXBOT V3 (prototipo visual/overlay)

Este repositorio implementa un prototipo **experimental** de análisis visual sobre ruleta con overlay y telemetría.

## Estado real

- ✅ Calibración de ROI para captura de pantalla.
- ✅ Detección visual de bola (brillo/saturación) y marcador de rotor (HSV verde).
- ✅ Modelo físico mejorado: ajuste local de dinámica (`dω/dt = -k_lin*ω - k_coulomb*sign(ω)`), integración temporal y proyección de impacto.
- ✅ Capa heurística de dispersión por deflectores + métrica de confianza.
- ✅ Overlay con sector, ángulo, telemetría y puntos de debug.
- ✅ Logging útil en consola.
- ⚠️ Sigue siendo un prototipo experimental (sensibilidad a iluminación/stream y dinámica no totalmente observable).

## Estructura final

```text
ALEXBOT/
├── main.py
├── run_demo.py
├── calibrate.py
├── requirements.txt
├── engine/
│   ├── vision.py
│   └── physics.py
├── ui/
│   └── overlay.py
├── utils/
│   ├── config.py
│   └── mapping.py
├── tests/
│   ├── test_config.py
│   ├── test_mapping.py
│   └── test_physics.py
└── tools/
    └── generate_demo_video.py
```

# ALEXBOT V3 PRO (Experimental Visual Predictor)

Proyecto experimental de visión por computador para estimar el ángulo de una bola y el marcador de referencia del rotor dentro de una ROI calibrada.

## Estructura

```text
/ALEXBOT
│   main.py
│   calibrate.py
│   requirements.txt
│   config.json (autogenerado)
├── engine
│   ├── vision.py
│   └── physics.py
├── ui
│   └── overlay.py
└── utils
    └── mapping.py
```

## Flujo de ejecución

1. Ejecuta `calibrate.py` para generar `config.json` con la ROI.
2. Ejecuta `main.py`.
3. El sistema:
   - captura video de la ROI,
   - detecta bola y marcador verde del rotor,
   - estima cinemática (omega/alpha) para la bola,
   - proyecta ángulo relativo bola-vs-rotor,
   - muestra el resultado en un overlay.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Modelo físico actual (resumen)

1. Estima `ω` instantánea con diferencias angulares envueltas.
2. Ajusta una regresión local `ω(t)` para derivar `α` efectiva.
3. Identifica coeficientes de fricción lineal + Coulomb.
4. Integra la dinámica hasta umbral de caída para obtener `t_drop`.
5. Proyecta ángulo de impacto y aplica una dispersión heurística por deflectores.

## Modos de ejecución (demo reproducible)

### 1) Demo sobre video local

1. Genera video sintético (opcional, reproducible):
   ```bash
   python tools/generate_demo_video.py --output demo.mp4
   ```
2. Ejecuta la demo:
   ```bash
   python main.py --source video --video-path demo.mp4 --roi-manual 0,0,800,800 --overlay
   ```

> Usa `--max-frames 300` para pruebas rápidas.

### 2) Demo sobre captura de pantalla

1. Calibra ROI:
   ```bash
   python calibrate.py
   ```
2. Ejecuta:
   ```bash
   python main.py --source screen --config config.json --overlay
   ```

### 3) Demo con overlay + logs

```bash
python main.py --source video --video-path demo.mp4 --roi-manual 0,0,800,800 --log-level INFO --overlay
```

## Script de arranque simple

```bash
python run_demo.py --mode video --video-path demo.mp4 --roi-manual 0,0,800,800 --overlay
```

## Teclas

- `q` o `ESC`: salir.

## Tests básicos

```bash
pytest -q
```

## Limitaciones pendientes

- La detección depende de iluminación/compresión/calidad del stream.
- El componente de deflectores es heurístico y requiere ajuste por rueda/dealer para mejorar calibración.
- Ningún sistema de visión 2D garantiza precisión universal en todas las mesas y condiciones de streaming.
   python main.py --source screen --config config.json
   ```

### 3) Demo con overlay + logs

```bash
python main.py --source video --video-path demo.mp4 --roi-manual 0,0,800,800 --log-level INFO
```

## Script de arranque simple

```bash
python run_demo.py --mode video --video-path demo.mp4 --roi-manual 0,0,800,800
```

## Teclas

- `q` o `ESC`: salir.

## Tests básicos

```bash
pytest -q
```

## Limitaciones pendientes

- La detección depende de iluminación/compresión/calidad del stream.
- El modelo de caída no incluye rebotes/deflectores/interacciones complejas.
- El sistema debe considerarse herramienta de análisis experimental, no predictor robusto garantizado.
## Uso

```bash
python3 calibrate.py
python3 main.py
```

## Limitaciones reales

- No garantiza precisión estable entre mesas/cámaras/iluminación.
- El modelo de caída es una aproximación cinemática, no un simulador físico completo.
- Requiere recalibración y ajuste por entorno.

## Aviso

Este repositorio se mantiene como experimento técnico de visión + señal temporal. No incluye automatización de apuestas.
