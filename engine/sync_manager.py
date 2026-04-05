"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class SyncTelemetry:
    """Datos de telemetría de sincronización."""
    drift_ms: float
    dynamic_offset_ms: float
    aligned_timestamp: float
    confidence: float


class SyncManager:
    """Sincronizador de Lag: Compensa el drift temporal entre evento visual
    y confirmación del servidor.
    
    Implementa auto-calibración adaptativa basada en mediciones históricas
    del lag de red.
    """

    def __init__(self, offset: float = 1.5, smoothing: float = 0.2, 
                 history_size: int = 10):
        """Inicializa el sincronizador.
        
        Args:
            offset: Offset inicial de drift en segundos (ej. 1.5s para lag de video)
            smoothing: Factor de suavizado EMA (0.0-1.0, mayor = más reactivo)
            history_size: Tamaño del historial de mediciones
        """
        self.drift = float(offset)
        self.smoothing = float(np.clip(smoothing, 0.01, 1.0)) if np is not None else 0.2
        self._history: deque[float] = deque(maxlen=history_size)
        self._last_update: Optional[float] = None
        self._calibration_count = 0
        self._min_drift = 0.1
        self._max_drift = 5.0

    def get_real_time_prediction(self, prediction_time: float) -> float:
        """Compensa el lag del video para obtener tiempo real.
        
        Args:
            prediction_time: Timestamp de la predicción capturada
            
        Returns:
            Timestamp ajustado restando el drift de red
        """
        return float(prediction_time) - self.drift

    def update_drift(self, visual_impact_t: float, server_result_t: float) -> float:
        """Auto-calibración: compara impacto visual vs publicación del número.
        
        Args:
            visual_impact_t: Timestamp cuando la bola impactó visualmente
            server_result_t: Timestamp cuando el servidor publicó el resultado
            
        Returns:
            Nuevo valor de drift calculado
        """
        # Calcular nuevo drift basado en la diferencia
        new_drift = float(server_result_t) - float(visual_impact_t)
        
        # Validar rango razonable
        new_drift = max(self._min_drift, min(self._max_drift, new_drift))
        
        # Suavizado exponencial (EMA)
        self.drift = (self.drift * (1.0 - self.smoothing)) + (new_drift * self.smoothing)
        
        # Guardar en historial
        self._history.append(self.drift)
        self._last_update = time.time()
        self._calibration_count += 1
        
        return self.drift

    def update_drift_batch(self, measurements: List[Tuple[float, float]]) -> float:
        """Actualiza el drift con múltiples mediciones.
        
        Args:
            measurements: Lista de tuplas (visual_impact_t, server_result_t)
            
        Returns:
            Nuevo valor de drift promedio
        """
        if not measurements:
            return self.drift
        
        drifts = []
        for visual_t, server_t in measurements:
            new_drift = float(server_t) - float(visual_t)
            new_drift = max(self._min_drift, min(self._max_drift, new_drift))
            drifts.append(new_drift)
        
        if np is not None:
            median_drift = float(np.median(drifts))
        else:
            median_drift = float(sorted(drifts)[len(drifts) // 2])
        
        # Aplicar suavizado
        self.drift = (self.drift * (1.0 - self.smoothing)) + (median_drift * self.smoothing)
        self._history.append(self.drift)
        self._last_update = time.time()
        self._calibration_count += len(measurements)
        
        return self.drift

    def get_sync_telemetry(self, visual_ts: float, confirmation_ts: float) -> SyncTelemetry:
        """Calcula métricas completas de sincronización.
        
        Args:
            visual_ts: Timestamp del evento visual
            confirmation_ts: Timestamp de confirmación
            
        Returns:
            SyncTelemetry con métricas de drift
        """
        drift_ms = (float(confirmation_ts) - float(visual_ts)) * 1000.0
        aligned = self.align_timestamp(visual_ts)
        
        # Calcular confianza basada en la estabilidad del historial
        confidence = self._calculate_confidence()
        
        return SyncTelemetry(
            drift_ms=drift_ms,
            dynamic_offset_ms=self.drift * 1000.0,
            aligned_timestamp=aligned,
            confidence=confidence
        )

    def align_timestamp(self, visual_ts: float) -> float:
        """Alinea un timestamp visual al tiempo real compensado.
        
        Args:
            visual_ts: Timestamp visual original
            
        Returns:
            Timestamp alineado
        """
        return float(visual_ts) + self.drift

    def _calculate_confidence(self) -> float:
        """Calcula confianza basada en la estabilidad del drift.
        
        Returns:
            Valor de confianza entre 0.0 y 1.0
        """
        if len(self._history) < 3:
            return 0.5
        
        if np is not None:
            std_dev = float(np.std(list(self._history)))
            # Menor desviación estándar = mayor confianza
            confidence = max(0.0, min(1.0, 1.0 - (std_dev / self.drift)))
        else:
            confidence = 0.7
        
        return confidence

    def get_stats(self) -> dict:
        """Obtiene estadísticas del sincronizador.
        
        Returns:
            Diccionario con estadísticas de drift
        """
        stats = {
            'current_drift': self.drift,
            'calibration_count': self._calibration_count,
            'last_update': self._last_update,
        }
        
        if len(self._history) > 0:
            history_list = list(self._history)
            if np is not None:
                stats.update({
                    'drift_mean': float(np.mean(history_list)),
                    'drift_std': float(np.std(history_list)),
                    'drift_min': float(np.min(history_list)),
                    'drift_max': float(np.max(history_list)),
                })
            else:
                stats.update({
                    'drift_mean': sum(history_list) / len(history_list),
                    'drift_min': min(history_list),
                    'drift_max': max(history_list),
                })
        
        return stats

    def reset(self, offset: Optional[float] = None) -> None:
        """Reinicia el sincronizador.
        
        Args:
            offset: Nuevo offset inicial (opcional)
        """
        if offset is not None:
            self.drift = float(offset)
        self._history.clear()
        self._last_update = None
        self._calibration_count = 0

    def adjust_for_network_type(self, network_type: str) -> None:
        """Ajusta parámetros según el tipo de red.
        
        Args:
            network_type: 'fiber', 'cable', 'wifi', 'mobile'
        """
        adjustments = {
            'fiber': {'offset': 1.2, 'smoothing': 0.15},
            'cable': {'offset': 1.5, 'smoothing': 0.2},
            'wifi': {'offset': 1.8, 'smoothing': 0.25},
            'mobile': {'offset': 2.2, 'smoothing': 0.3},
        }
        
        config = adjustments.get(network_type, adjustments['cable'])
        self.drift = config['offset']
        self.smoothing = config['smoothing']
