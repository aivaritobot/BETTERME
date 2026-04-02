import cv2
import numpy as np
from mss import mss

class RouletteEye:
    def __init__(self, x, y, w, h):
        # Define el área de la pantalla que el bot va a observar
        self.monitor = {"top": y, "left": x, "width": w, "height": h}
        self.sct = mss()

    def get_ball(self):
        # Toma una captura de pantalla del área de la ruleta
        img = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Filtro para detectar la bola blanca
        lower = np.array([0, 0, 200])
        upper = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                # Retorna las coordenadas X y Y de la bola
                return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None
