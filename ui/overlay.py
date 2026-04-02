from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

import importlib


def _load_cv2():
    try:
        return importlib.import_module("cv2")
    except Exception:  # pragma: no cover
        return None


def announce_text(text: str) -> None:
    try:
        import pyttsx3

        tts = pyttsx3.init()
        tts.say(text)
        tts.runAndWait()
    except Exception:
        return


def render_stealth_overlay(
    frame,
    wheel_center,
    wheel_radius,
    confidence: float,
    edge: float,
    top_numbers: list,
    legal_warning: bool = True,
):
    cv2 = _load_cv2()
    if cv2 is None or frame is None:
        return

    if wheel_center and wheel_radius:
        cv2.circle(frame, wheel_center, wheel_radius, (80, 80, 80), 1)

    # HUD verde solo cuando confianza Shannon > 0.82 (señal real)
    hud_color = (0, 220, 100) if confidence > 0.82 else (180, 180, 180)
    info = f"C:{confidence:.2f} E:{edge:.2f} Top:{top_numbers[:3]}"
    cv2.putText(frame, info, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hud_color, 1)

    if legal_warning:
        cv2.putText(
            frame, "EXPERIMENTAL/RESEARCH ONLY",
            (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 255), 1,
        )

    cv2.imshow("BETTERME BESTIA", frame)
    cv2.waitKey(1)


def render_live_overlay(*args, **kwargs):
    frame = kwargs.get("frame") if kwargs else (args[0] if args else None)
    confidence = kwargs.get("confidence", 0.0)
    edge = kwargs.get("edge", 0.0)
    sector = kwargs.get("sector", [])
    render_stealth_overlay(
        frame=frame,
        wheel_center=kwargs.get("wheel_center"),
        wheel_radius=kwargs.get("wheel_radius"),
        confidence=confidence,
        edge=edge,
        top_numbers=sector,
    )


def render_green_overlay(frame, recommended_zone: str, confidence: float):
    render_stealth_overlay(frame, None, None, confidence, 0.0, [recommended_zone])
