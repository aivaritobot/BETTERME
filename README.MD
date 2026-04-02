# BETTERME (Experimental Live Roulette Visual Assistant)

Prototipo de visión por computador para estimar zona probable de caída en ruleta y **mostrar sugerencias** en pantalla (y opcionalmente voz).  
**No automatiza apuestas.**

## ⚠️ Advertencia legal y ética

Este software es **experimental**, **no garantiza ganancias** y su uso puede ser **ilegal o violar términos de servicio** en casinos físicos/u online. Úsalo solo en contextos de laboratorio/investigación y bajo tu responsabilidad.

## Mejoras implementadas

- Entrada en vivo por webcam (`--source 0`) o stream RTSP/archivo (`--source rtsp://...`).
- Auto-detección de rueda por `HoughCircles` (sin ROI manual obligatoria).
- Detección de bola/marcador con YOLOv8 (si hay modelo) + fallback robusto por visión clásica.
- Tracking con filtro de Kalman para estabilizar detecciones entre frames.
- Auto-calibración de fricción (`k_linear`, `k_coulomb`) desde spins en vivo y guardado en `config.json`.
- Física mejorada: integración numérica con `scipy.integrate.odeint`, ruido por rebotes/deflectores y sector de 8-12 números según dispersión.
- Overlay mejorado: confianza, sugerencia, sector, edge y precisión histórica.
- Voz opcional con `pyttsx3`.

## Estructura

- `main.py` → orquestador en vivo + sugerencias.
- `engine/vision.py` → captura, detección rueda/objetos, tracking.
- `engine/physics.py` → modelo físico, calibración, confianza, sugerencias.
- `ui/overlay.py` → visualización en pantalla.

## Instalación (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dependencias clave

- `opencv-python`
- `numpy`
- `scipy`
- `ultralytics`
- `pyttsx3`

## Uso

### Webcam
```bash
python main.py --source 0 --bankroll 200 --confidence-threshold 0.70
```

### RTSP
```bash
python main.py --source "rtsp://usuario:pass@host:554/stream" --voice
```

### YOLO personalizado
```bash
python main.py --source 0 --yolo-model models/roulette_yolov8.pt
```

## Notas

- Para voz en Linux puede requerirse `espeak`.
- El valor de confianza es probabilístico y sensible a calidad de imagen.
