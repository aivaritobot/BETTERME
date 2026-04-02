from __future__ import annotations

import importlib


def _load_cv2():
    try:
        return importlib.import_module('cv2')
    except Exception:  # pragma: no cover
        return None


def _zone_rect(frame_shape, zone: str) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = frame_shape[:2]
    if zone == 'Docena 1':
        return (int(w * 0.05), int(h * 0.70)), (int(w * 0.30), int(h * 0.95))
    if zone == 'Docena 2':
        return (int(w * 0.35), int(h * 0.70)), (int(w * 0.60), int(h * 0.95))
    if zone == 'Docena 3':
        return (int(w * 0.65), int(h * 0.70)), (int(w * 0.90), int(h * 0.95))
    if zone == 'Rojo':
        return (int(w * 0.08), int(h * 0.55)), (int(w * 0.32), int(h * 0.67))
    if zone == 'Negro':
        return (int(w * 0.68), int(h * 0.55)), (int(w * 0.92), int(h * 0.67))
    return (int(w * 0.40), int(h * 0.40)), (int(w * 0.60), int(h * 0.60))


def render_green_overlay(frame, recommended_zone: str, confidence: float):
    cv2 = _load_cv2()
    if cv2 is None or frame is None:  # pragma: no cover
        return

    (x1, y1), (x2, y2) = _zone_rect(frame.shape, recommended_zone)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(frame, f'ALTA PROBABILIDAD: {recommended_zone}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f'Confianza: {confidence:.2%}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)
    cv2.imshow('ALEXBOT OVERLAY', frame)
    cv2.waitKey(1)
