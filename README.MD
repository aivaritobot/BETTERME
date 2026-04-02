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
