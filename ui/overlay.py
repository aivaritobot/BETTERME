from __future__ import annotations

import cv2
import numpy as np

from engine.physics import Prediction
from utils.mapping import get_alexbot_sector


def show_alex_overlay(frame, relative_prediction, telemetry: Prediction | None = None, debug=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (620, 180), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if relative_prediction is not None:
        name, numbers = get_alexbot_sector(relative_prediction)
        cv2.putText(frame, f'SECTOR: {name}', (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'NUMS: {numbers[:8]}', (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(frame, f'ANGLE: {relative_prediction:06.2f}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        cx, cy = w // 2, h // 2
        rad = np.radians(relative_prediction)
        px, py = int(cx + 100 * np.cos(rad)), int(cy + 100 * np.sin(rad))
        cv2.arrowedLine(frame, (cx, cy), (px, py), (0, 255, 0), 3, tipLength=0.25)

    if telemetry:
        cv2.putText(frame, f'omega={telemetry.ball_omega:7.2f}', (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 255), 1)
        cv2.putText(frame, f'alpha={telemetry.ball_alpha:7.2f}', (190, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 255), 1)
        cv2.putText(frame, f't_drop={telemetry.t_drop:4.2f}s', (360, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 255), 1)
        cv2.putText(frame, f'impact={telemetry.impact_angle:06.2f}', (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1)
        cv2.putText(frame, f'conf={telemetry.confidence:.2f}', (220, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1)
        if telemetry.experimental:
            cv2.putText(frame, 'MODELO EXPERIMENTAL', (360, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

    if debug:
        center = debug.get('center')
        ball_center = debug.get('ball_center')
        rotor_center = debug.get('rotor_center')
        status = debug.get('status', 'ok')
        cv2.putText(frame, f'status={status}', (360, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if center:
            cv2.circle(frame, center, 4, (255, 255, 0), -1)
        if ball_center:
            cv2.circle(frame, ball_center, 5, (255, 255, 255), 2)
        if rotor_center:
            cv2.circle(frame, rotor_center, 5, (0, 255, 0), 2)

    cv2.imshow('ALEXBOT V3 - ANALISIS EXPERIMENTAL', frame)
