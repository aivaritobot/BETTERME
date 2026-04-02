# Audit Report (2026-04-02)

Este reporte documenta verificación real de ejecutabilidad y consistencia del repositorio.

## Resultados

- ✅ Sintaxis/compilación Python: `py_compile` sobre módulos principales.
- ✅ Tests automáticos: `pytest -q`.
- ✅ CLI help funcional en entorno headless: `python main.py --help`.
- ✅ Generación de demo reproducible: `python tools/generate_demo_video.py ...`.
- ✅ Ejecución E2E en modo video (sin overlay): `python main.py --source video ... --max-frames 20`.

## Evidencia de corrección de hallazgos del audit previo

1. **"physics.py IndentationError"** → no reproducido; compilación OK.
2. **"mapping.py SyntaxError"** → no reproducido; compilación OK.
3. **"tests no corren"** → corregido; suite pasando.
4. **"main.py --help rompe por libGL"** → corregido con imports lazy + `opencv-python-headless`.

## Estado técnico actual

- El proyecto corre extremo a extremo en modo video reproducible.
- Soporta modo screen con calibración y modo overlay opcional (GUI-capable).
- El modelo físico es avanzado respecto a la versión anterior, pero mantiene incertidumbre inherente de observación 2D.

## Comando único de auditoría

```bash
python tools/run_audit.py
```
