from __future__ import annotations

import importlib


def _load_cv2():
    try:
        return importlib.import_module("cv2")
    except Exception:  # pragma: no cover
        return None


def render_live_overlay(
    frame,
    wheel_center,
    wheel_radius,
    ball_center,
    marker_center,
    confidence: float,
    suggestion_text: str,
    sector: list,
    historical_accuracy: float,
    edge: float,
):
    cv2 = _load_cv2()
    if cv2 is None or frame is None:  # pragma: no cover
        return

    if wheel_center and wheel_radius:
        cv2.circle(frame, wheel_center, wheel_radius, (60, 180, 60), 2)

    if ball_center:
        cv2.circle(frame, ball_center, 7, (255, 255, 255), -1)
        if wheel_center:
            cv2.arrowedLine(frame, wheel_center, ball_center, (230, 230, 230), 2)

    if marker_center:
        cv2.circle(frame, marker_center, 7, (0, 255, 0), -1)
        if wheel_center:
            cv2.arrowedLine(frame, wheel_center, marker_center, (0, 200, 0), 2)

    sector_label = f"{sector[0]}-{sector[-1]}" if sector else "N/A"
    color = (0, 220, 0) if confidence >= 0.70 else (0, 190, 255)

    cv2.putText(frame, f"Confianza: {confidence:.1%}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Apuesta: {suggestion_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Sector: {sector_label}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 255, 220), 2)
    cv2.putText(frame, f"Precision hist: {historical_accuracy:.1%} | Edge: {edge:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 2)

    cv2.imshow("BETTERME Live Assistant", frame)
    cv2.waitKey(1)


def render_green_overlay(frame, recommended_zone: str, confidence: float):
    """Compat legacy."""
    render_live_overlay(
        frame=frame,
        wheel_center=None,
        wheel_radius=None,
        ball_center=None,
        marker_center=None,
        confidence=confidence,
        suggestion_text=f"{recommended_zone}",
        sector=[],
        historical_accuracy=0.0,
        edge=0.0,
    )
