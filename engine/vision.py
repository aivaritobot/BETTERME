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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ball_angle, ball_center = self._detect_ball_angle(hsv)
        zero_angle, zero_center = self._detect_zero_marker_angle(hsv)

        debug = {
            'ball_center': ball_center,
            'zero_center': zero_center,
            'center': self.center,
        }
        return ball_angle, zero_angle, frame, debug

    def _detect_ball_angle(self, hsv):
        # Detección robusta de zonas brillantes: canal V alto y baja saturación
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        bright = cv2.adaptiveThreshold(
            v,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -12,
        )
        low_sat = cv2.inRange(s, 0, 70)
        mask = cv2.bitwise_and(bright, low_sat)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        center = self._largest_contour_center(mask, min_area=8)
        if center is None:
            return None, None
        return self._polar_angle(center), center

    def _detect_zero_marker_angle(self, hsv):
        # Verde de referencia del rotor (rango amplio para variaciones de mesa/stream)
        lower = np.array([35, 70, 60], dtype=np.uint8)
        upper = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        center = self._largest_contour_center(mask, min_area=15)
        if center is None:
            return None, None
        return self._polar_angle(center), center

    def _largest_contour_center(self, mask, min_area=10):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            return None

        moments = cv2.moments(c)
        if moments['m00'] <= 0:
            return None

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy

    def _polar_angle(self, point):
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return float(np.degrees(np.arctan2(dy, dx)) % 360)
