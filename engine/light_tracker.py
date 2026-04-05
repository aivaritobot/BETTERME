"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import mss
except Exception:  # pragma: no cover
    mss = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class BallPosition:
    """Resultado de detección de posición de la bola."""
    x: float
    y: float
    timestamp: float
    confidence: float


class LightTracker:
    """El 'Ojo' Rápido: Tracker de baja latencia para detección de bola.
    
    Utiliza captura de pantalla rápida de una región de interés (ROI)
    y procesamiento numpy optimizado para detectar el punto más brillante
    (la bola) en tiempo real.
    """

    def __init__(self, config: dict):
        """Inicializa el tracker con configuración.
        
        Args:
            config: Diccionario con 'roi_coordinates' y 'pixel_threshold'
        """
        self.config = config
        self.sct = mss.mss() if mss is not None else None
        self.roi = config.get('roi_coordinates', {'x': 100, 'y': 100, 'width': 600, 'height': 600})
        self.threshold = config.get('pixel_threshold', 240)
        self._last_position: Optional[Tuple[float, float]] = None
        self._last_timestamp: Optional[float] = None

    def _get_roi_dict(self) -> dict:
        """Convierte las coordenadas ROI al formato que espera mss."""
        return {
            'left': int(self.roi.get('x', 100)),
            'top': int(self.roi.get('y', 100)),
            'width': int(self.roi.get('width', 600)),
            'height': int(self.roi.get('height', 600))
        }

    def get_ball_position(self) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """Captura y analiza la ROI para encontrar la posición de la bola.
        
        Returns:
            Tuple de ((x, y), timestamp) o (None, None) si no se detecta
        """
        if self.sct is None or np is None:
            return None, None

        try:
            # Captura rápida de la región de interés (ROI)
            screenshot = self.sct.grab(self._get_roi_dict())
            img = np.array(screenshot)
            
            # Convertir a escala de grises para velocidad (usar canal rojo es más rápido)
            gray = img[:, :, 0]
            
            # Encontrar el punto más brillante (la bola)
            indices = np.where(gray >= self.threshold)
            
            if len(indices[0]) > 0:
                # Calcular centroide de los píxeles brillantes
                y_mean = float(np.mean(indices[0]))
                x_mean = float(np.mean(indices[1]))
                
                # Ajustar coordenadas relativas a la pantalla completa
                x_absolute = x_mean + self.roi.get('x', 100)
                y_absolute = y_mean + self.roi.get('y', 100)
                
                timestamp = time.time()
                self._last_position = (x_absolute, y_absolute)
                self._last_timestamp = timestamp
                
                return (x_absolute, y_absolute), timestamp
            
            return None, None
            
        except Exception:
            return None, None

    def get_ball_position_enhanced(self) -> Optional[BallPosition]:
        """Versión mejorada que retorna un objeto BallPosition completo.
        
        Returns:
            BallPosition con coordenadas, timestamp y confianza, o None
        """
        pos, ts = self.get_ball_position()
        if pos is None or ts is None:
            return None
        
        # Calcular confianza basada en la cantidad de píxeles detectados
        if self.sct is not None and np is not None:
            try:
                screenshot = self.sct.grab(self._get_roi_dict())
                img = np.array(screenshot)
                gray = img[:, :, 0]
                bright_pixels = np.sum(gray >= self.threshold)
                # Normalizar confianza: más píxeles = más confianza (hasta un límite)
                confidence = min(1.0, bright_pixels / 100.0)
            except Exception:
                confidence = 0.5
        else:
            confidence = 0.5
        
        return BallPosition(x=pos[0], y=pos[1], timestamp=ts, confidence=confidence)

    def calibrate_threshold(self, target_brightness: int = 240) -> None:
        """Auto-calibra el umbral basado en el brillo actual.
        
        Args:
            target_brightness: Valor objetivo de brillo (0-255)
        """
        if self.sct is None or np is None:
            return
        
        try:
            screenshot = self.sct.grab(self._get_roi_dict())
            img = np.array(screenshot)
            gray = img[:, :, 0]
            
            # Encontrar el percentil 95 del brillo
            p95 = np.percentile(gray, 95)
            
            # Ajustar umbral para alcanzar el brillo objetivo
            self.threshold = int(min(255, max(200, p95 * 0.9)))
            
        except Exception:
            self.threshold = target_brightness

    def close(self) -> None:
        """Libera recursos del tracker."""
        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass
            self.sct = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
