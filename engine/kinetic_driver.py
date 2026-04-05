"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida."""

from __future__ import annotations

import random
import time
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable

try:  # pragma: no cover
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class MovementResult:
    """Resultado de un movimiento ejecutado."""
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    path_points: List[Tuple[int, int]]
    duration_ms: float
    executed: bool
    reason: str


def bezier_quadratic(p0: float, p1: float, p2: float, t: float) -> float:
    """Fórmula de Bézier cuadrática para movimiento fluido.
    
    Args:
        p0: Punto inicial
        p1: Punto de control
        p2: Punto final
        t: Parámetro de interpolación (0.0 a 1.0)
        
    Returns:
        Valor interpolado en el punto t
    """
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def bezier_cubic(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Fórmula de Bézier cúbica para movimiento más natural.
    
    Args:
        p0: Punto inicial
        p1: Primer punto de control
        p2: Segundo punto de control
        p3: Punto final
        t: Parámetro de interpolación (0.0 a 1.0)
        
    Returns:
        Valor interpolado en el punto t
    """
    return ((1 - t) ** 3 * p0 + 
            3 * (1 - t) ** 2 * t * p1 + 
            3 * (1 - t) * t ** 2 * p2 + 
            t ** 3 * p3)


class KineticDriver:
    """El 'Brazo' Humano: Driver de movimiento con comportamiento antropomórfico.
    
    Simula movimientos humanos usando curvas de Bézier, jitter natural,
    y variabilidad en velocidad para evitar patrones robóticos.
    """

    def __init__(self, 
                 bezier_steps: int = 15,
                 jitter_px: int = 3,
                 click_delay_ms: float = 50,
                 micro_delay_ms: Tuple[int, int] = (1, 9)):
        """Inicializa el driver cinético.
        
        Args:
            bezier_steps: Número de pasos en la curva de Bézier
            jitter_px: Desviación máxima en píxeles para simular error humano
            click_delay_ms: Delay antes del clic en milisegundos
            micro_delay_ms: Rango de delay entre movimientos (min, max)
        """
        self.bezier_steps = max(5, bezier_steps)
        self.jitter_px = max(0, jitter_px)
        self.click_delay_ms = max(0, click_delay_ms)
        self.micro_delay_ms = micro_delay_ms
        self._enabled = pyautogui is not None
        self._movement_history: List[dict] = []
        self._rng = random.Random()
        
        # Configurar pyautogui para velocidad
        if pyautogui is not None:
            pyautogui.MINIMUM_DURATION = 0
            pyautogui.MINIMUM_SLEEP = 0
            pyautogui.PAUSE = 0

    def _generate_control_points(self, 
                                  start: Tuple[int, int], 
                                  end: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Genera puntos de control aleatorios para la curva de Bézier.
        
        Args:
            start: Posición inicial (x, y)
            end: Posición final (x, y)
            
        Returns:
            Tupla de dos puntos de control (cp1, cp2)
        """
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Variación aleatoria para simular inconsistencia humana
        offset_range = 50
        cp1_x = mid_x + self._rng.randint(-offset_range, offset_range)
        cp1_y = start[1] + self._rng.randint(-offset_range, 0)  # Tendencia hacia arriba
        
        cp2_x = mid_x + self._rng.randint(-offset_range // 2, offset_range // 2)
        cp2_y = mid_y + self._rng.randint(-offset_range // 2, offset_range // 2)
        
        return (int(cp1_x), int(cp1_y)), (int(cp2_x), int(cp2_y))

    def _generate_path(self, 
                       start: Tuple[int, int], 
                       end: Tuple[int, int],
                       steps: Optional[int] = None) -> List[Tuple[int, int]]:
        """Genera una trayectoria de Bézier entre dos puntos.
        
        Args:
            start: Posición inicial (x, y)
            end: Posición final (x, y)
            steps: Número de pasos (usa self.bezier_steps si es None)
            
        Returns:
            Lista de puntos (x, y) formando la trayectoria
        """
        steps = steps or self.bezier_steps
        cp1, cp2 = self._generate_control_points(start, end)
        
        path = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 1.0
            
            # Usar Bézier cúbica para movimiento más natural
            x = bezier_cubic(start[0], cp1[0], cp2[0], end[0], t)
            y = bezier_cubic(start[1], cp1[1], cp2[1], end[1], t)
            
            path.append((int(round(x)), int(round(y))))
        
        return path

    def _apply_jitter(self, pos: Tuple[int, int], jitter: Optional[int] = None) -> Tuple[int, int]:
        """Aplica jitter aleatorio a una posición.
        
        Args:
            pos: Posición original (x, y)
            jitter: Magnitud del jitter (usa self.jitter_px si es None)
            
        Returns:
            Posición con jitter aplicado
        """
        jitter = jitter if jitter is not None else self.jitter_px
        
        if jitter <= 0:
            return pos
        
        # Distribución gaussiana para jitter más realista
        if np is not None:
            jx = int(np.random.normal(0, jitter / 2))
            jy = int(np.random.normal(0, jitter / 2))
        else:
            jx = self._rng.randint(-jitter, jitter)
            jy = self._rng.randint(-jitter, jitter)
        
        return (pos[0] + jx, pos[1] + jy)

    def move_to(self, 
                x: int, 
                y: int, 
                duration: Optional[float] = None,
                jitter: Optional[int] = None) -> MovementResult:
        """Mueve el cursor a una posición con trayectoria Bézier.
        
        Args:
            x: Coordenada X destino
            y: Coordenada Y destino
            duration: Duración total del movimiento (None = automático)
            jitter: Override del jitter por defecto
            
        Returns:
            MovementResult con detalles del movimiento
        """
        if not self._enabled or pyautogui is None:
            return MovementResult(
                start_pos=(0, 0),
                end_pos=(x, y),
                path_points=[],
                duration_ms=0,
                executed=False,
                reason="pyautogui no disponible"
            )
        
        start_pos = pyautogui.position()
        start_tuple = (int(start_pos.x), int(start_pos.y))
        end_tuple = (int(x), int(y))
        
        # Generar trayectoria
        path = self._generate_path(start_tuple, end_tuple)
        
        # Calcular duración basada en distancia si no se especifica
        if duration is None:
            distance = math.sqrt((end_tuple[0] - start_tuple[0])**2 + 
                               (end_tuple[1] - start_tuple[1])**2)
            # Velocidad humana promedio: ~1000-2000 px/segundo
            duration = max(0.1, min(0.5, distance / 1500))
        
        step_delay = duration / len(path) if path else 0
        
        start_time = time.perf_counter()
        
        # Ejecutar movimiento paso a paso
        for point in path:
            jittered = self._apply_jitter(point, jitter)
            pyautogui.moveTo(jittered[0], jittered[1], duration=0)
            
            # Micro-delay entre pasos para simular movimiento humano
            if self.micro_delay_ms[1] > 0:
                micro = self._rng.randint(self.micro_delay_ms[0], self.micro_delay_ms[1]) / 1000.0
                time.sleep(micro)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        result = MovementResult(
            start_pos=start_tuple,
            end_pos=end_tuple,
            path_points=path,
            duration_ms=duration_ms,
            executed=True,
            reason="ok"
        )
        
        self._movement_history.append({
            'timestamp': time.time(),
            'result': result
        })
        
        # Mantener historial limitado
        if len(self._movement_history) > 100:
            self._movement_history.pop(0)
        
        return result

    def click(self, 
              x: Optional[int] = None, 
              y: Optional[int] = None,
              jitter: Optional[int] = None,
              button: str = 'left') -> bool:
        """Realiza un clic en la posición especificada.
        
        Args:
            x: Coordenada X (None = posición actual)
            y: Coordenada Y (None = posición actual)
            jitter: Override del jitter por defecto
            button: Botón del mouse ('left', 'right', 'middle')
            
        Returns:
            True si el clic se ejecutó correctamente
        """
        if not self._enabled or pyautogui is None:
            return False
        
        try:
            # Mover primero si se especificó posición
            if x is not None and y is not None:
                self.move_to(x, y, jitter=jitter)
            
            # Delay antes del clic (reacción humana)
            if self.click_delay_ms > 0:
                # Añadir variabilidad al delay
                delay_variation = self._rng.randint(-10, 20)
                actual_delay = max(0, self.click_delay_ms + delay_variation) / 1000.0
                time.sleep(actual_delay)
            
            # Ejecutar clic con jitter final
            final_jitter = jitter if jitter is not None else self.jitter_px
            if final_jitter > 0:
                current = pyautogui.position()
                jittered = self._apply_jitter((int(current.x), int(current.y)), final_jitter)
                pyautogui.click(jittered[0], jittered[1], button=button)
            else:
                pyautogui.click(button=button)
            
            return True
            
        except Exception:
            return False

    def move_and_click(self, 
                       x: int, 
                       y: int, 
                       jitter: Optional[int] = None,
                       pre_click_delay: Optional[float] = None) -> MovementResult:
        """Mueve y hace clic en una posición con comportamiento humano completo.
        
        Args:
            x: Coordenada X destino
            y: Coordenada Y destino
            jitter: Desviación máxima en píxeles para error humano
            pre_click_delay: Delay adicional antes del clic
            
        Returns:
            MovementResult con detalles del movimiento
        """
        # Ejecutar movimiento
        result = self.move_to(x, y, jitter=jitter)
        
        if not result.executed:
            return result
        
        # Delay adicional opcional
        if pre_click_delay is not None:
            time.sleep(pre_click_delay)
        
        # Ejecutar clic
        click_success = self.click(jitter=jitter)
        
        if not click_success:
            result.reason = "movimiento ok, clic fallido"
        
        return result

    def simulate_hover(self, 
                       x: int, 
                       y: int, 
                       duration_ms: float = 500) -> bool:
        """Simula un hover (pasar el mouse por encima sin clic).
        
        Args:
            x: Coordenada X destino
            y: Coordenada Y destino
            duration_ms: Duración del hover en milisegundos
            
        Returns:
            True si se ejecutó correctamente
        """
        if not self._enabled or pyautogui is None:
            return False
        
        result = self.move_to(x, y)
        if result.executed:
            time.sleep(duration_ms / 1000.0)
            return True
        return False

    def get_position(self) -> Tuple[int, int]:
        """Obtiene la posición actual del cursor.
        
        Returns:
            Tupla (x, y) con la posición actual
        """
        if pyautogui is not None:
            pos = pyautogui.position()
            return (int(pos.x), int(pos.y))
        return (0, 0)

    def set_enabled(self, enabled: bool) -> None:
        """Habilita o deshabilita la ejecución real de movimientos.
        
        Args:
            enabled: True para habilitar, False para modo simulación
        """
        self._enabled = enabled and pyautogui is not None

    def get_stats(self) -> dict:
        """Obtiene estadísticas de uso del driver.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'total_movements': len(self._movement_history),
            'enabled': self._enabled,
            'bezier_steps': self.bezier_steps,
            'jitter_px': self.jitter_px,
            'avg_duration_ms': (
                sum(m['result'].duration_ms for m in self._movement_history) / 
                len(self._movement_history) if self._movement_history else 0
            )
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False
