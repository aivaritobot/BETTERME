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


def _draw_entropy_bar(frame, entropy: float) -> None:
    """MEJORA GOD: barra visual de incertidumbre (entropía normalizada)."""
    cv2 = _load_cv2()
    if cv2 is None:
        return
    entropy = float(min(1.0, max(0.0, entropy)))
    x, y, w, h = 12, 54, 180, 12
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), 1)
    fill = int(w * (1.0 - entropy))
    color = (0, 210, 100) if entropy < 0.35 else (0, 180, 220) if entropy < 0.55 else (70, 70, 255)
    cv2.rectangle(frame, (x + 1, y + 1), (x + fill, y + h - 1), color, -1)
    cv2.putText(frame, f"Uncertainty:{entropy:.2f}", (x + w + 8, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)


def render_stealth_overlay(
    frame,
    wheel_center,
    wheel_radius,
    confidence: float,
    edge: float,
    top_numbers: list,
    entropy: float = 1.0,
    prediction: dict | None = None,
    legal_warning: bool = True,
):
    cv2 = _load_cv2()
    if cv2 is None or frame is None:
        return

    if wheel_center and wheel_radius:
        cv2.circle(frame, wheel_center, wheel_radius, (80, 80, 80), 1)

    hud_color = (0, 220, 100) if confidence > 0.82 else (180, 180, 180)
    info = f"C:{confidence:.2f} E:{edge:.2f} H:{entropy:.2f} Top:{top_numbers[:3]}"
    cv2.putText(frame, info, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hud_color, 1)

    # === GOD SINGLE NUMBER MODE - AÑADIDO ===
    if prediction and prediction.get("display_text"):
        if prediction.get("mode") == "single_god":
            # === MAX LEVEL ONLINE GOD MODE - AÑADIDO ===
            cv2.rectangle(frame, (40, 90), (760, 230), (0, 255, 0), 2)
            cv2.putText(
                frame,
                prediction["display_text"],
                (60, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.8,
                prediction.get("color", (0, 255, 0)),
                5,
            )
            cv2.putText(
                frame,
                f"Prob: {prediction.get('max_prob', 0.0)*100:.1f}% | Edge: {prediction.get('edge', 0.0):.1%}",
                (60, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                prediction["display_text"],
                (60, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.9,
                prediction.get("color", (0, 255, 255)),
                3,
            )

    if legal_warning:
        cv2.putText(frame, "EXPERIMENTAL/RESEARCH ONLY", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 255), 1)

    _draw_entropy_bar(frame, entropy)

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
        entropy=kwargs.get("entropy", 1.0),
    )


def render_green_overlay(frame, recommended_zone: str, confidence: float):
    render_stealth_overlay(frame, None, None, confidence, 0.0, [recommended_zone])
