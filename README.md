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
