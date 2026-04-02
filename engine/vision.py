import cv2
import numpy as np
import mss

class AlexBotVision:
    def __init__(self, roi):
        self.sct = mss.mss()
        self.roi = roi 
        self.center = (roi['width'] // 2, roi['height'] // 2)

    def get_alex_data(self):
        img = np.array(self.sct.grab(self.roi))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Filtro de brillo ALEXBOT para la bola
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        _, mask = cv2.threshold(v_channel, 215, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Cálculo Polar ALEXBOT respecto al eje central
                dx, dy = cx - self.center[0], cy - self.center[1]
                angle = np.degrees(np.atan2(dy, dx)) % 360
                return angle, frame
        return None, frame
