import cv2
import numpy as np
import mss

class VisionEngine:
    def __init__(self, monitor_area):
        self.sct = mss.mss()
        self.monitor = monitor_area # {'top': 100, 'left': 100, 'width': 500, 'height': 500}
        
    def get_positions(self):
        img = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Tracking de la bola (Blanco puro)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ball_mask = cv2.inRange(hsv, (0, 0, 220), (180, 40, 255))
        
        # Tracking del Cero (Verde neón del rotor)
        zero_mask = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))
        
        # Extraer coordenadas (X, Y)
        ball_coords = self._get_center(ball_mask)
        zero_coords = self._get_center(zero_mask)
        
        return ball_coords, zero_coords, frame

    def _get_center(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            M = cv2.moments(max(cnts, key=cv2.contourArea))
            if M["m00"] != 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
