from __future__ import annotations

import cv2
import mss
import numpy as np

from utils.config import DetectionConfig, ROI


class AlexBotVision:
    """Pipeline de adquisición + detección para modo video o pantalla."""

    def __init__(
        self,
        roi: ROI,
        source: str = 'screen',
        video_path: str | None = None,
        detection_config: DetectionConfig | None = None,
    ):
        self.roi = roi
        self.source = source
        self.detection = detection_config or DetectionConfig()
        self.center = (roi.width // 2, roi.height // 2)

        self.sct = None
        self.cap = None
        if source == 'screen':
            self.sct = mss.mss()
        elif source == 'video':
            if not video_path:
                raise ValueError('video_path es requerido cuando source=video')
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f'No se pudo abrir video: {video_path}')
        else:
            raise ValueError("source debe ser 'screen' o 'video'")

    def close(self):
        if self.cap is not None:
            self.cap.release()

    def get_alex_data(self):
        frame = self._next_frame()
        if frame is None:
            return None, None, None, {'status': 'eof'}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ball_angle, ball_center = self._detect_ball_angle(hsv)
        rotor_angle, rotor_center = self._detect_rotor_angle(hsv)

        debug = {
            'ball_center': ball_center,
            'rotor_center': rotor_center,
            'center': self.center,
            'status': 'ok',
        }
        return ball_angle, rotor_angle, frame, debug

    def _next_frame(self):
        if self.source == 'screen':
            img = np.array(
                self.sct.grab(
                    {
                        'top': self.roi.top,
                        'left': self.roi.left,
                        'width': self.roi.width,
                        'height': self.roi.height,
                    }
                )
            )
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        ok, frame = self.cap.read()
        if not ok:
            return None

        h, w = frame.shape[:2]
        top = max(0, min(self.roi.top, h - 1))
        left = max(0, min(self.roi.left, w - 1))
        bottom = max(top + 1, min(top + self.roi.height, h))
        right = max(left + 1, min(left + self.roi.width, w))
        return frame[top:bottom, left:right]

    def _detect_ball_angle(self, hsv):
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]

        block_size = self.detection.ball_block_size
        if block_size % 2 == 0:
            block_size += 1

        bright = cv2.adaptiveThreshold(
            v,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.detection.ball_c_offset,
        )
        low_sat = cv2.inRange(s, 0, self.detection.ball_sat_max)
        mask = cv2.bitwise_and(bright, low_sat)
        mask = self._clean_mask(mask)

        center = self._largest_contour_center(mask, self.detection.ball_min_area)
        if center is None:
            return None, None
        return self._polar_angle(center), center

    def _detect_rotor_angle(self, hsv):
        lower = np.array(self.detection.rotor_hsv_lower, dtype=np.uint8)
        upper = np.array(self.detection.rotor_hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = self._clean_mask(mask)

        center = self._largest_contour_center(mask, self.detection.rotor_min_area)
        if center is None:
            return None, None
        return self._polar_angle(center), center

    @staticmethod
    def _clean_mask(mask):
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    @staticmethod
    def _largest_contour_center(mask, min_area=10):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            return None

        moments = cv2.moments(c)
        if moments['m00'] <= 0:
            return None

        return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])

    def _polar_angle(self, point):
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return float(np.degrees(np.arctan2(dy, dx)) % 360)
