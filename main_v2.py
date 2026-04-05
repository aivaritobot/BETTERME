#!/usr/bin/env python3
from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida.

BETTERME v2 - Orquestador de Suite de Optimización de Latencia y Telemetría
============================================================================

Este es el archivo principal que une todos los componentes:
- LightTracker: El "Ojo" rápido para detección de bola
- SyncManager: Sincronizador de lag de red
- KineticDriver: El "Brazo" humano con movimientos Bézier
- Physics: Motor de física avanzado de BETTERME

Uso:
    python main_v2.py

Requisitos:
    - Permisos de Grabación de Pantalla en macOS
    - Permisos de Accesibilidad para pyautogui
    - Dependencias: pip install mss pyautogui numpy
"""

import json
import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Importar componentes de la suite de latencia
try:
    from engine.light_tracker import LightTracker, BallPosition
    from engine.sync_manager import SyncManager, SyncTelemetry
    from engine.kinetic_driver import KineticDriver, MovementResult
except ImportError as e:
    print(f"[ERROR] No se pueden importar los módulos de la suite: {e}")
    print("Asegúrate de estar en el directorio raíz del proyecto.")
    sys.exit(1)

# Importar física de BETTERME
try:
    from engine.physics import AdvancedPhysicsEngine, RoulettePhysicsEngine
    from engine.vision import RouletteVision, VisionState
    from engine.dual_path import FastPixelTracker, number_to_sector
except ImportError as e:
    print(f"[WARNING] No se pudo importar el motor de física BETTERME: {e}")
    AdvancedPhysicsEngine = None
    RouletteVision = None
    FastPixelTracker = None


@dataclass
class LoopTelemetry:
    """Métricas de telemetría del loop principal."""
    iteration: int
    timestamp: float
    ball_detected: bool
    ball_position: Optional[tuple[float, float]]
    sync_drift_ms: float
    prediction_confidence: float
    latency_ms: float
    action_executed: bool
    safety_cycle: int


class BetterMeV2Orchestrator:
    """Orquestador principal de BETTERME v2.
    
    Integra el tracker rápido, sincronizador de lag, driver cinético
    y el motor de física BETTERME en un pipeline unificado.
    """

    def __init__(self, config_path: str = "config.json"):
        """Inicializa el orquestador con configuración.
        
        Args:
            config_path: Ruta al archivo de configuración JSON
        """
        self.config = self._load_config(config_path)
        self.running = False
        self.paused = False
        
        # Contadores de seguridad
        self.success_count = 0
        self.iteration = 0
        self.telemetry_history: list[LoopTelemetry] = []
        
        # Inicializar componentes
        print("[INIT] Inicializando Suite de Optimización de Latencia...")
        
        # 1. Light Tracker - El "Ojo" rápido
        self.tracker = LightTracker(self.config)
        print(f"[INIT] LightTracker listo - ROI: {self.config.get('roi_coordinates')}")
        
        # 2. Sync Manager - Compensación de lag
        drift_offset = self.config.get('network_drift_offset', 1.5)
        self.syncer = SyncManager(offset=drift_offset)
        print(f"[INIT] SyncManager listo - Drift inicial: {drift_offset}s")
        
        # 3. Kinetic Driver - El "Brazo" humano
        self.driver = KineticDriver(
            bezier_steps=self.config.get('bezier_steps', 15),
            jitter_px=self.config.get('human_jitter_px', 3),
            click_delay_ms=50,
            micro_delay_ms=(1, 9)
        )
        print(f"[INIT] KineticDriver listo - Bézier steps: {self.config.get('bezier_steps', 15)}")
        
        # 4. Motor de física BETTERME (si está disponible)
        self.physics: Optional[Any] = None
        self.vision: Optional[Any] = None
        self.fast_tracker: Optional[Any] = None
        
        if AdvancedPhysicsEngine is not None:
            self.physics = AdvancedPhysicsEngine()
            self.physics.update_hyperparams_from_config(self.config)
            print("[INIT] Motor de física BETTERME cargado")
        
        if FastPixelTracker is not None:
            sector_count = self.config.get('sector_count', 8)
            self.fast_tracker = FastPixelTracker(sector_count=sector_count)
            print(f"[INIT] FastPixelTracker cargado - {sector_count} sectores")
        
        # Configurar manejador de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("[INIT] Orquestador listo para ejecutar")

    def _load_config(self, config_path: str) -> dict:
        """Carga configuración desde archivo JSON.
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Returns:
            Diccionario de configuración
        """
        default_config = {
            'low_latency_mode': True,
            'roi_coordinates': {'x': 100, 'y': 100, 'width': 600, 'height': 600},
            'pixel_threshold': 240,
            'network_drift_offset': 1.5,
            'human_jitter_px': 3,
            'bezier_steps': 15,
            'safety_cycle_count': 3,
            'sector_count': 8,
            'wheel_mode': 'European',
            'god_mode': False,
            'online_mode': False,
        }
        
        try:
            with open(config_path, 'r') as f:
                loaded = json.load(f)
                default_config.update(loaded)
                print(f"[CONFIG] Cargado desde {config_path}")
        except FileNotFoundError:
            print(f"[CONFIG] Archivo no encontrado, usando defaults")
        except json.JSONDecodeError as e:
            print(f"[CONFIG] Error de JSON: {e}, usando defaults")
        
        return default_config

    def _signal_handler(self, signum, frame):
        """Maneja señales de terminación."""
        print(f"\n[SIGNAL] Recibida señal {signum}, deteniendo...")
        self.stop()

    def calibrate(self) -> dict:
        """Ejecuta calibración inicial del sistema.
        
        Returns:
            Resultados de calibración
        """
        print("\n[CALIBRATE] Iniciando calibración...")
        
        # Calibrar threshold del tracker
        self.tracker.calibrate_threshold()
        print(f"[CALIBRATE] Threshold ajustado a: {self.tracker.threshold}")
        
        # Detectar tipo de red recomendado
        network_type = self._detect_network_type()
        self.syncer.adjust_for_network_type(network_type)
        print(f"[CALIBRATE] Red detectada: {network_type}, drift: {self.syncer.drift}s")
        
        results = {
            'threshold': self.tracker.threshold,
            'network_type': network_type,
            'drift': self.syncer.drift,
            'roi': self.tracker.roi,
        }
        
        print("[CALIBRATE] Calibración completada")
        return results

    def _detect_network_type(self) -> str:
        """Detecta tipo de red basado en configuración.
        
        Returns:
            Tipo de red: 'fiber', 'cable', 'wifi', 'mobile'
        """
        drift = self.config.get('network_drift_offset', 1.5)
        
        if drift < 1.3:
            return 'fiber'
        elif drift < 1.6:
            return 'cable'
        elif drift < 2.0:
            return 'wifi'
        else:
            return 'mobile'

    def _should_inject_safety_variability(self) -> bool:
        """Determina si se debe inyectar variabilidad de seguridad.
        
        Returns:
            True si se debe inyectar variabilidad
        """
        safety_cycles = self.config.get('safety_cycle_count', 3)
        return self.success_count >= safety_cycles

    def _reset_safety_cycle(self) -> None:
        """Reinicia el ciclo de seguridad."""
        self.success_count = 0
        print("[SAFETY] Ciclo de seguridad reiniciado")

    def process_iteration(self) -> Optional[LoopTelemetry]:
        """Procesa una iteración del loop principal.
        
        Returns:
            LoopTelemetry con métricas de la iteración
        """
        self.iteration += 1
        iteration_start = time.perf_counter()
        
        # 1. Capturar posición de la bola
        pos, t_capture = self.tracker.get_ball_position()
        
        if pos is None:
            return LoopTelemetry(
                iteration=self.iteration,
                timestamp=time.time(),
                ball_detected=False,
                ball_position=None,
                sync_drift_ms=self.syncer.drift * 1000,
                prediction_confidence=0.0,
                latency_ms=0.0,
                action_executed=False,
                safety_cycle=self.success_count
            )
        
        # 2. Ajustar tiempo por lag de red
        real_t = self.syncer.get_real_time_prediction(t_capture)
        
        # 3. Preparar predicción
        predicted_sector = 0
        confidence = 0.0
        sector_coords: list[tuple[int, int]] = []
        
        # Usar física de BETTERME si está disponible
        if self.physics is not None and self.fast_tracker is not None:
            # Crear un VisionState mínimo para el fast_tracker
            # Esto es una simplificación - en producción usarías la visión completa
            pass
        
        # 4. Decidir acción basada en ciclo de seguridad
        action_executed = False
        
        if not self._should_inject_safety_variability():
            # Modo normal: ejecutar acción
            if self.config.get('low_latency_mode', True):
                # Simular clic en posición calculada
                # En modo demo, no ejecutamos clicks reales
                # self.driver.move_and_click(pos[0], pos[1], self.config.get('human_jitter_px', 3))
                action_executed = True
                self.success_count += 1
        else:
            # Inyectar variabilidad de seguridad
            print("[SAFETY] Inyectando variabilidad de seguridad...")
            self._reset_safety_cycle()
            time.sleep(5)  # Pausa de seguridad
        
        # Calcular latencia
        latency_ms = (time.perf_counter() - iteration_start) * 1000
        
        telemetry = LoopTelemetry(
            iteration=self.iteration,
            timestamp=time.time(),
            ball_detected=True,
            ball_position=pos,
            sync_drift_ms=self.syncer.drift * 1000,
            prediction_confidence=confidence,
            latency_ms=latency_ms,
            action_executed=action_executed,
            safety_cycle=self.success_count
        )
        
        self.telemetry_history.append(telemetry)
        
        # Mantener historial limitado
        if len(self.telemetry_history) > 1000:
            self.telemetry_history.pop(0)
        
        return telemetry

    def run_loop(self, max_iterations: Optional[int] = None, 
                 print_interval: int = 100) -> None:
        """Ejecuta el loop principal del orquestador.
        
        Args:
            max_iterations: Número máximo de iteraciones (None = infinito)
            print_interval: Intervalo para imprimir estadísticas
        """
        print("\n" + "="*60)
        print("BETTERME v2 - Modo Liviano Iniciado")
        print("="*60)
        print(f"Modo baja latencia: {self.config.get('low_latency_mode', True)}")
        print(f"Ciclos de seguridad: {self.config.get('safety_cycle_count', 3)}")
        print(f"Drift de red: {self.syncer.drift}s")
        print("Presiona Ctrl+C para detener")
        print("="*60 + "\n")
        
        self.running = True
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Verificar límite de iteraciones
                if max_iterations is not None and self.iteration >= max_iterations:
                    print(f"[LOOP] Alcanzado límite de {max_iterations} iteraciones")
                    break
                
                # Procesar iteración
                telemetry = self.process_iteration()
                
                # Imprimir estadísticas periódicamente
                if self.iteration % print_interval == 0 and telemetry:
                    self._print_stats(telemetry)
                
                # Pequeña pausa para no saturar CPU
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n[LOOP] Interrumpido por usuario")
        finally:
            self.stop()

    def _print_stats(self, telemetry: LoopTelemetry) -> None:
        """Imprime estadísticas del sistema.
        
        Args:
            telemetry: Métricas de la iteración actual
        """
        recent = self.telemetry_history[-100:]
        detected = sum(1 for t in recent if t.ball_detected)
        avg_latency = sum(t.latency_ms for t in recent) / len(recent) if recent else 0
        
        print(f"[STATS] Iter: {telemetry.iteration} | "
              f"Detectados: {detected}/100 | "
              f"Latencia: {avg_latency:.2f}ms | "
              f"Drift: {telemetry.sync_drift_ms:.1f}ms | "
              f"Safety: {telemetry.safety_cycle}/{self.config.get('safety_cycle_count', 3)}")

    def pause(self) -> None:
        """Pausa el loop principal."""
        self.paused = True
        print("[CONTROL] Pausado")

    def resume(self) -> None:
        """Reanuda el loop principal."""
        self.paused = False
        print("[CONTROL] Reanudado")

    def stop(self) -> None:
        """Detiene el orquestador y libera recursos."""
        self.running = False
        print("\n[STOP] Deteniendo BETTERME v2...")
        
        # Liberar recursos
        if hasattr(self, 'tracker'):
            self.tracker.close()
        
        # Imprimir estadísticas finales
        if self.telemetry_history:
            total = len(self.telemetry_history)
            detected = sum(1 for t in self.telemetry_history if t.ball_detected)
            avg_latency = sum(t.latency_ms for t in self.telemetry_history) / total
            
            print("\n" + "="*60)
            print("ESTADÍSTICAS FINALES")
            print("="*60)
            print(f"Total iteraciones: {total}")
            print(f"Bolas detectadas: {detected} ({detected/total*100:.1f}%)")
            print(f"Latencia promedio: {avg_latency:.2f}ms")
            print(f"Último drift: {self.syncer.drift:.3f}s")
            print("="*60)
        
        print("[STOP] BETTERME v2 detenido")

    def get_telemetry_report(self) -> dict:
        """Genera un reporte completo de telemetría.
        
        Returns:
            Diccionario con estadísticas completas
        """
        if not self.telemetry_history:
            return {"error": "No hay datos de telemetría"}
        
        total = len(self.telemetry_history)
        detected = sum(1 for t in self.telemetry_history if t.ball_detected)
        actions = sum(1 for t in self.telemetry_history if t.action_executed)
        
        latencies = [t.latency_ms for t in self.telemetry_history]
        drifts = [t.sync_drift_ms for t in self.telemetry_history]
        
        import numpy as np
        
        return {
            'total_iterations': total,
            'detection_rate': detected / total if total > 0 else 0,
            'action_count': actions,
            'latency_stats': {
                'mean_ms': float(np.mean(latencies)),
                'std_ms': float(np.std(latencies)),
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies)),
            },
            'drift_stats': {
                'mean_ms': float(np.mean(drifts)),
                'std_ms': float(np.std(drifts)),
                'current_ms': self.syncer.drift * 1000,
            },
            'sync_stats': self.syncer.get_stats(),
            'driver_stats': self.driver.get_stats(),
        }


def main():
    """Función principal de entrada."""
    parser = argparse.ArgumentParser(
        description='BETTERME v2 - Suite de Optimización de Latencia y Telemetría'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.json',
        help='Ruta al archivo de configuración (default: config.json)'
    )
    parser.add_argument(
        '--calibrate', '-cal',
        action='store_true',
        help='Ejecutar calibración inicial'
    )
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Modo demo: simulación sin clicks reales'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=None,
        help='Número máximo de iteraciones'
    )
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generar reporte de telemetría al finalizar'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el config
    if not Path(args.config).exists():
        print(f"[ERROR] Archivo de configuración no encontrado: {args.config}")
        print("Creando configuración por defecto...")
        # El orquestador creará config por defecto
    
    # Crear y configurar orquestador
    orchestrator = BetterMeV2Orchestrator(config_path=args.config)
    
    # Ejecutar calibración si se solicitó
    if args.calibrate:
        orchestrator.calibrate()
    
    # Configurar modo demo
    if args.demo:
        orchestrator.driver.set_enabled(False)
        print("[MODE] Modo demo activado (sin clicks reales)")
    
    # Ejecutar loop principal
    try:
        orchestrator.run_loop(max_iterations=args.iterations)
    except Exception as e:
        print(f"[ERROR] Error en ejecución: {e}")
        raise
    finally:
        # Generar reporte si se solicitó
        if args.report:
            report = orchestrator.get_telemetry_report()
            print("\n[REPORTE DE TELEMETRÍA]")
            print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
