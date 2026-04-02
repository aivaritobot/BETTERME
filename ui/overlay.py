import cv2
import numpy as np
from utils.mapping import get_alexbot_sector

def show_alex_overlay(frame, prediction):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if prediction:
        name, numbers = get_alexbot_sector(prediction)
        cv2.putText(frame, f"ALEXBOT SECTOR: {name}", (20, 45), 1, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"INFO: {str(numbers[:6])}...", (20, 85), 1, 0.9, (200, 200, 200), 1)
        
        # Guía Visual ALEXBOT
        cx, cy = w//2, h//2
        rad = np.radians(prediction)
        px, py = int(cx + 85*np.cos(rad)), int(cy + 85*np.sin(rad))
        cv2.arrowedLine(frame, (cx, cy), (px, py), (0, 255, 0), 3, tipLength=0.3)

    cv2.imshow("ALEXBOT V3 - PREDICTOR PRO", frame)
    cv2.waitKey(1)
