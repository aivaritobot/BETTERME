# BETTERME BESTIA (Experimental)

**EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida.**

Proyecto de investigación para modelado físico-estocástico y visión computacional en ruleta.

## Núcleo físico (Small & Tse + híbrido)

Ecuación usada en `engine/physics.py`:

\[
\frac{d^2\theta_b}{dt^2} = -\mu g \operatorname{sign}(\omega_b) - \beta \omega_b^2 - \gamma (\omega_b - \omega_w)
\]

- fricción lineal/coulomb + arrastre cuadrático + acoplamiento rotor.
- integración numérica con `scipy.odeint`.
- ruido estocástico (deflectores + pockets) sobre distribución final de 37 números.
- detección de tilt/bias por concentración sectorial.
- auto-calibración cada 5 spins con `LinearRegression` + esquema bayesiano ligero y filtro de partículas (`filterpy`, si está disponible).

## Visión

- YOLO (`yolov11n.pt`) o modelo custom para bola/marcadores.
- fallback con `HoughCircles` para rueda.
- homografía automática para corrección de perspectiva.
- estructura preparada para tracking multiobjeto.

## Predicción y riesgo

- Distribución de probabilidad en 37 números.
- Confianza real:

\[
\text{confidence} = 1 - \left(\frac{\sigma}{360}\right) \cdot \text{factor\_tilt}
\]

- Edge:

\[
\text{edge} = (P_{hit} - 1/37) \cdot \text{payout\_neto}
\]

- Kelly fraccional conservador:

\[
\text{bet} = bankroll \cdot \left(\frac{edge}{variance}\right) \cdot 0.5
\]

Señal solo si `edge > 0.12` y `confidence > 0.68`.

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Modo principal

```bash
python main.py --source 0 --yolo-model yolov11n.pt --bankroll 100
```

### Solo auditoría (sin overlay)

```bash
python main.py --audit-only --source 0
```

### Simulación Monte Carlo (10k runs)

```bash
python main.py --simulate
# o
python tools/monte_carlo_sim.py
```

## Auditami Bot (Telegram)

```bash
python auditami_bot.py
```

Comandos:
- `/report`
- `/bias`
- `/log`

## Nota de investigación

En escenarios favorables de simulación estocástica, el sistema puede observar rangos de +$180 a +$480 en 1 hora desde $100 inicial, con varianza alta y sin garantía.

## GOD MODE Features

> Todas las mejoras son **opt-in** para mantener compatibilidad total.

- **MEJORA GOD (Visión):** YOLO custom vía `--yolo-model`, tracking EKF, fallback híbrido YOLO + clásico + optical-flow (Farneback), detección de fase (`high_speed`, `decelerating`, `dropping`) y homografía auto con más puntos.
- **MEJORA GOD (Tracking):** estado expandido `θ, ω, α, friction`, ruido adaptativo por fase, fallback multiobjeto con ByteTrack cuando hay oclusiones.
- **MEJORA GOD (Física):** híbrido físico + red residual (`engine/hybrid_physics.py`), Monte Carlo avanzado (hasta 2000 sims), entropía de Shannon, wheel-bias adaptativo, fórmula multifactor de confianza.
- **MEJORA GOD (Decisión):** señal fuerte solo cuando `edge > 0.15`, `confidence > 0.78` y entropía baja.
- **MEJORA GOD (UI):** barra de incertidumbre/entropía en overlay.
- **MEJORA GOD (Logs):** cada spin queda en `logs/spins/*.json`.

### Comandos GOD MODE

```bash
# Compatibilidad total (comportamiento legacy)
python main.py --source 0

# GOD MODE completo
python main.py \
  --source 0 \
  --god-mode \
  --yolo-model yolov11n.pt \
  --mc-sims 1500 \
  --bankroll 100

# Modo granular (opt-in por componente)
python main.py \
  --source 0 \
  --use-ekf \
  --hybrid-detection \
  --hybrid-physics \
  --multi-object-fallback \
  --yolo-conf-threshold 0.75
```

### Entrenar YOLO custom

```bash
# 1) prepara dataset en formato YOLO (ball / marker)
# 2) entrena (ejemplo ultralytics)
yolo detect train data=roulette.yaml model=yolo11n.pt epochs=100 imgsz=960 batch=16

# 3) exporta para inferencia rápida
yolo export model=runs/detect/train/weights/best.pt format=onnx
# opcional TensorRT
yolo export model=runs/detect/train/weights/best.pt format=engine

# 4) usa el modelo en BETTERME
python main.py --yolo-model runs/detect/train/weights/best.pt --god-mode
```

### Entrenar red residual (Hybrid Physics)

1. Ejecuta sesiones y acumula spins en `logs/spins/*.json`.
2. Genera dataset supervisado (`pred_angle`, `real_angle`, `omega_b`, `omega_w`).
3. Invoca `AdvancedPhysicsEngine.train_hybrid_from_spins(...)` para entrenar.
4. El modelo se guarda en `models/hybrid_residual.pt` y se carga automáticamente si `--hybrid-physics` o `--god-mode` está activo.

### Nota de rendimiento

Para latencia baja y FPS altos, exporta YOLO a ONNX/TensorRT y usa resolución ajustada al ROI de rueda.
