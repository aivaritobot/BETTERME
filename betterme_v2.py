#!/usr/bin/env python3
from __future__ import annotations

"""EXPERIMENTAL. Solo investigación. Ilegal en la mayoría de casinos. Riesgo total de pérdida.

BETTERME V2.0: SISTEMA UNIFICADO DE PREDICCIÓN Y EJECUCIÓN
===========================================================

Sistema integrado con interfaz gráfica flotante para:
- Tracking rápido de bola (Light Mode)
- Cálculo de velocidad angular
- Ejecución con curvas Bézier humanizadas
- Telemetría en tiempo real

Uso:
    python betterme_v2.py

Requisitos:
    pip install mss pyautogui numpy opencv-python
    
Permisos macOS:
    - Accesibilidad
    - Grabación de Pantalla
"""

import tkinter as tk
import json
import os
import time
import random
import math
from threading import Thread
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    import mss
except Exception:
    mss = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import pyautogui
except Exception:
    pyautogui = None  # type: ignore


@dataclass
class BallDetection:
    """Resultado de detección de bola."""
    x: float
    y: float
    timestamp: float
    confidence: float


@dataclass
class PhysicsPrediction:
    """Resultado de predicción física."""
    sector: int
    omega: float
    confidence: float
    predicted_number: int


# --- MÓDULOS DE INGENIERÍA ---

class BETTERME_Core:
    """Núcleo del sistema BETTERME v2.0."""

    def __init__(self, config_path: str = 'engine/config.json'):
        self.config_path = config_path
        self.load_config()
        self.sct = mss.mss() if mss is not None else None
        self.active = False
        self.detection_history: list[BallDetection] = []
        self.last_omega = 0.0
        self.last_angle = 0.0
        
        # Centros para cálculo angular (se actualizan con calibración)
        self.wheel_center: Optional[Tuple[int, int]] = None
        self.wheel_radius: Optional[int] = None

    def load_config(self) -> None:
        """Carga configuración desde archivo o crea defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception:
                self._set_default_config()
        else:
            self._set_default_config()
            self.save_config()

    def _set_default_config(self) -> None:
        """Establece configuración por defecto."""
        self.config = {
            "roi_coordinates": {"left": 0, "top": 0, "width": 400, "height": 400},
            "pixel_threshold": 240,
            "network_drift": 1.5,
            "human_jitter": 3,
            "safety_cycle": 3,
            "bezier_steps": 15,
            "wheel_center": None,
            "wheel_radius": None,
            "monte_carlo_sims": 500,
            "dispersion_std_deg": 14.0,
            "wheel_mode": "European"
        }

    def save_config(self) -> None:
        """Guarda configuración actual a archivo."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"[ERROR] No se pudo guardar config: {e}")

    def detect_ball(self) -> Optional[BallDetection]:
        """Detecta la bola en la ROI usando luminancia."""
        if self.sct is None or np is None:
            return None

        try:
            roi = self.config['roi_coordinates']
            screenshot = np.array(self.sct.grab(roi))
            
            # Usar canal B (suficiente para detectar blanco/brillante)
            gray = screenshot[:, :, 0]
            
            # Detección de luminancia
            thresh = self.config['pixel_threshold']
            pts = np.where(gray >= thresh)
            
            if len(pts[0]) > 0:
                ball_x = float(np.mean(pts[1])) + roi['left']
                ball_y = float(np.mean(pts[0])) + roi['top']
                
                # Calcular confianza basada en cantidad de píxeles
                confidence = min(1.0, len(pts[0]) / 100.0)
                
                return BallDetection(
                    x=ball_x,
                    y=ball_y,
                    timestamp=time.time(),
                    confidence=confidence
                )
            
            return None
            
        except Exception as e:
            return None

    def calculate_angular_velocity(self, detection: BallDetection) -> float:
        """Calcula velocidad angular basada en posición actual e histórica."""
        if not self.detection_history:
            self.detection_history.append(detection)
            return 0.0

        last = self.detection_history[-1]
        dt = detection.timestamp - last.timestamp
        
        if dt < 0.001:
            return self.last_omega

        # Calcular ángulos respecto al centro de la rueda
        center = self.wheel_center
        if center is None:
            # Usar centro de ROI como aproximación
            roi = self.config['roi_coordinates']
            center = (roi['left'] + roi['width'] // 2, 
                     roi['top'] + roi['height'] // 2)

        # Ángulos en grados
        prev_angle = math.degrees(math.atan2(last.y - center[1], last.x - center[0]))
        curr_angle = math.degrees(math.atan2(detection.y - center[1], detection.x - center[0]))
        
        # Delta angular (manejo de wrap-around)
        delta_angle = ((curr_angle - prev_angle + 180) % 360) - 180
        omega = delta_angle / dt

        # Suavizado exponencial
        alpha = 0.7
        omega_smooth = alpha * omega + (1 - alpha) * self.last_omega
        
        self.last_omega = omega_smooth
        self.last_angle = curr_angle
        self.detection_history.append(detection)
        
        # Mantener historial limitado
        if len(self.detection_history) > 50:
            self.detection_history.pop(0)

        return omega_smooth

    def predict_sector(self, omega: float, detection: BallDetection) -> PhysicsPrediction:
        """Predice sector basado en velocidad angular y física."""
        # Física simplificada: estimar tiempo de caída
        # En una implementación completa, esto usaría el motor de física BETTERME
        
        if abs(omega) < 1.0:
            # Bola casi detenida
            confidence = 0.9
            predicted_number = self._angle_to_number(self.last_angle)
        else:
            # Estimar tiempo hasta caída
            t_drop = abs(omega) / 50.0  # Simplificación
            t_drop = min(3.0, max(0.5, t_drop))
            
            # Ángulo predicho
            predicted_angle = (self.last_angle + omega * t_drop) % 360
            predicted_number = self._angle_to_number(predicted_angle)
            
            # Confianza inversamente proporcional a la velocidad
            confidence = max(0.3, min(0.95, 1.0 - abs(omega) / 200.0))

        # Mapear número a sector (0-36)
        sector = predicted_number

        return PhysicsPrediction(
            sector=sector,
            omega=omega,
            confidence=confidence,
            predicted_number=predicted_number
        )

    def _angle_to_number(self, angle_deg: float) -> int:
        """Convierte ángulo a número de ruleta europea."""
        # Ruleta europea: 37 números (0-36)
        idx = int((angle_deg % 360) / 360.0 * 37) % 37
        
        # Orden de la ruleta europea
        european_wheel = [
            0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8,
            23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12,
            35, 3, 26
        ]
        
        return european_wheel[idx]

    def bezier_move(self, x: int, y: int) -> bool:
        """Mueve el ratón con curvas de Bézier para simular brazo humano."""
        if pyautogui is None:
            return False

        try:
            start_x, start_y = pyautogui.position()
            
            # Punto de control aleatorio para curva natural
            cp_x = (start_x + x) / 2 + random.randint(-50, 50)
            cp_y = (start_y + y) / 2 + random.randint(-50, 50)
            
            steps = self.config.get('bezier_steps', 15)
            
            for i in range(1, steps + 1):
                t = i / steps
                # Bézier cuadrática
                target_x = (1-t)**2 * start_x + 2*(1-t)*t * cp_x + t**2 * x
                target_y = (1-t)**2 * start_y + 2*(1-t)*t * cp_y + t**2 * y
                pyautogui.moveTo(int(target_x), int(target_y))
            
            # Clic con jitter humano
            jitter = self.config['human_jitter']
            final_x = x + random.randint(-jitter, jitter)
            final_y = y + random.randint(-jitter, jitter)
            pyautogui.click(final_x, final_y)
            
            return True
            
        except Exception:
            return False

    def calibrate_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Actualiza la región de interés."""
        self.config['roi_coordinates'] = {
            "left": x,
            "top": y,
            "width": width,
            "height": height
        }
        # Estimar centro de la rueda como centro de ROI
        self.wheel_center = (x + width // 2, y + height // 2)
        self.wheel_radius = min(width, height) // 2
        self.config['wheel_center'] = self.wheel_center
        self.config['wheel_radius'] = self.wheel_radius
        self.save_config()


# --- INTERFAZ DE USUARIO (OVERLAY) ---

class BetterMeApp:
    """Interfaz gráfica flotante del sistema."""

    def __init__(self, core: BETTERME_Core):
        self.core = core
        self.root = tk.Tk()
        self.root.title("BETTERME v2.0 - Tactical Overlay")
        self.root.attributes("-topmost", True, "-alpha", 0.9)
        self.root.geometry("520x700")
        self.root.configure(bg='#121212')
        
        # Variables de estado
        self.success_streak = 0
        self.total_detections = 0
        self.last_prediction: Optional[PhysicsPrediction] = None

        self._build_ui()
        self.log("Sistema BETTERME v2.0 Listo. Posiciona la ventana sobre la ruleta.")
        
        # Iniciar loop de actualización de UI
        self._schedule_ui_update()
        
        self.root.mainloop()

    def _build_ui(self) -> None:
        """Construye la interfaz de usuario."""
        # Header
        header = tk.Label(
            self.root, 
            text="BETTERME v2.0", 
            bg='#121212', 
            fg='#00ff00',
            font=("Consolas", 16, "bold")
        )
        header.pack(pady=10)

        # Frame de ROI visual
        roi_container = tk.Frame(self.root, bg='#121212')
        roi_container.pack(pady=10)
        
        self.roi_frame = tk.Canvas(
            roi_container, 
            width=450, 
            height=300, 
            bg="black", 
            highlightthickness=2, 
            highlightbackground="#00ff00"
        )
        self.roi_frame.pack()
        self.roi_frame.create_text(
            225, 150, 
            text="POSICIONA SOBRE RULETA\nLuego pulsa SINCRONIZAR", 
            fill="#333",
            font=("Consolas", 12)
        )

        # Indicador de estado
        self.status_label = tk.Label(
            self.root,
            text="ESTADO: STANDBY",
            bg='#121212',
            fg='#888888',
            font=("Consolas", 10)
        )
        self.status_label.pack(pady=5)

        # Stats frame
        stats_frame = tk.Frame(self.root, bg='#121212')
        stats_frame.pack(pady=5)
        
        self.stats_labels = {}
        stats = ["Detecciones:", "Racha:", "Omega:", "Confianza:"]
        for i, stat in enumerate(stats):
            lbl = tk.Label(stats_frame, text=stat, bg='#121212', fg='#00aa00', font=("Consolas", 9))
            lbl.grid(row=i, column=0, sticky='w', padx=5)
            val = tk.Label(stats_frame, text="--", bg='#121212', fg='#ffffff', font=("Consolas", 9))
            val.grid(row=i, column=1, sticky='w', padx=5)
            self.stats_labels[stat] = val

        # Chat / Telemetría
        console_frame = tk.Frame(self.root, bg='#121212')
        console_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        console_label = tk.Label(
            console_frame, 
            text="TELEMETRÍA", 
            bg='#121212', 
            fg='#00ff00',
            font=("Consolas", 10, "bold")
        )
        console_label.pack(anchor='w')

        self.console = tk.Text(
            console_frame, 
            height=12, 
            width=60, 
            bg="#000000", 
            fg="#00ff00", 
            font=("Consolas", 9),
            insertbackground='#00ff00'
        )
        self.console.pack(fill='both', expand=True)

        # Botones
        btn_frame = tk.Frame(self.root, bg='#121212')
        btn_frame.pack(pady=15)

        self.btn_sync = tk.Button(
            btn_frame, 
            text="📍 SINCRONIZAR ROI", 
            command=self.sync_roi, 
            bg="#00aa00", 
            fg="white", 
            font=("Arial", 10, "bold"),
            width=18,
            height=2
        )
        self.btn_sync.pack(side="left", padx=10)

        self.btn_run = tk.Button(
            btn_frame, 
            text="▶ INICIAR BOT", 
            command=self.toggle_bot, 
            bg="#ff0000", 
            fg="white", 
            font=("Arial", 10, "bold"),
            width=18,
            height=2
        )
        self.btn_run.pack(side="left", padx=10)

    def _schedule_ui_update(self) -> None:
        """Programa actualización periódica de la UI."""
        self._update_ui()
        self.root.after(100, self._schedule_ui_update)

    def _update_ui(self) -> None:
        """Actualiza indicadores de la UI."""
        if self.last_prediction:
            self.stats_labels["Omega:"].config(text=f"{self.last_prediction.omega:.2f}°/s")
            self.stats_labels["Confianza:"].config(text=f"{self.last_prediction.confidence:.2%}")
        
        self.stats_labels["Detecciones:"].config(text=str(self.total_detections))
        self.stats_labels["Racha:"].config(text=f"{self.success_streak}/{self.core.config['safety_cycle']}")

    def log(self, msg: str) -> None:
        """Agrega mensaje al console."""
        timestamp = time.strftime("%H:%M:%S")
        self.console.insert(tk.END, f"[{timestamp}] >>> {msg}\n")
        self.console.see(tk.END)
        # Limitar líneas
        if float(self.console.index('end')) > 100:
            self.console.delete('1.0', '10.0')

    def sync_roi(self) -> None:
        """Sincroniza la región de interés con la posición actual de la ventana."""
        # Obtener coordenadas absolutas del canvas ROI
        x = self.root.winfo_x() + self.roi_frame.winfo_x() + 2  # +2 por el borde
        y = self.root.winfo_y() + self.roi_frame.winfo_y() + 25  # +25 por el header de la ventana
        w = self.roi_frame.winfo_width() - 4  # -4 por los bordes
        h = self.roi_frame.winfo_height() - 4

        self.core.calibrate_roi(x, y, w, h)
        
        self.log(f"ROI Sincronizada: ({x}, {y}) [{w}x{h}]")
        self.log(f"Centro estimado: ({x + w//2}, {y + h//2})")
        
        # Actualizar visual
        self.roi_frame.delete("all")
        self.roi_frame.create_rectangle(2, 2, w, h, outline="#00ff00", width=2)
        self.roi_frame.create_text(
            w//2, h//2,
            text="ROI ACTIVA",
            fill="#00ff00",
            font=("Consolas", 14, "bold")
        )
        
        self.status_label.config(text="ESTADO: ROI SINCRONIZADA", fg="#00ff00")

    def toggle_bot(self) -> None:
        """Activa/desactiva el bot."""
        self.core.active = not self.core.active
        
        if self.core.active:
            self.btn_run.config(text="⏹ DETENER BOT", bg="#ff5500")
            self.status_label.config(text="ESTADO: ONLINE", fg="#00ff00")
            self.log("BOT ACTIVADO - Iniciando tracking...")
            Thread(target=self.main_loop, daemon=True).start()
        else:
            self.btn_run.config(text="▶ INICIAR BOT", bg="#ff0000")
            self.status_label.config(text="ESTADO: STANDBY", fg="#888888")
            self.log("BOT DETENIDO")

    def main_loop(self) -> None:
        """Loop principal de detección y predicción."""
        while self.core.active:
            try:
                # 1. TRACKING RÁPIDO (LIGHT MODE)
                detection = self.core.detect_ball()
                
                if detection:
                    self.total_detections += 1
                    
                    # 2. CÁLCULO DE VELOCIDAD ANGULAR
                    omega = self.core.calculate_angular_velocity(detection)
                    
                    # 3. PREDICCIÓN FÍSICA
                    prediction = self.core.predict_sector(omega, detection)
                    self.last_prediction = prediction
                    
                    # Compensar drift de red
                    t_real = detection.timestamp - self.core.config['network_drift']
                    
                    self.log(
                        f"BOLA #{self.total_detections} | "
                        f"Sector: {prediction.predicted_number} | "
                        f"ω: {omega:.1f}°/s | "
                        f"Conf: {prediction.confidence:.1%}"
                    )
                    
                    # 4. EJECUCIÓN (Si no estamos en ciclo de seguridad)
                    if self.success_streak < self.core.config['safety_cycle']:
                        # Aquí iría la ejecución real
                        # self.core.bezier_move(target_x, target_y)
                        self.success_streak += 1
                        self.log(f"Ejecución simulada (Racha: {self.success_streak})")
                        
                        # Cooldown por ronda
                        time.sleep(15)
                    else:
                        self.log("SEGURIDAD: Inyectando pérdida controlada...")
                        self.success_streak = 0
                        time.sleep(30)
                
                # Pequeña pausa para no saturar CPU (100 FPS check)
                time.sleep(0.01)
                
            except Exception as e:
                self.log(f"ERROR: {str(e)}")
                time.sleep(1)


def main():
    """Punto de entrada principal."""
    print("=" * 60)
    print("BETTERME v2.0 - Sistema Unificado de Predicción")
    print("=" * 60)
    print("AVISO: Solo para investigación. Uso ilegal en casinos.")
    print("=" * 60)
    
    # Verificar dependencias
    missing = []
    if mss is None:
        missing.append("mss")
    if np is None:
        missing.append("numpy")
    if pyautogui is None:
        missing.append("pyautogui")
    
    if missing:
        print(f"\n[ERROR] Faltan dependencias: {', '.join(missing)}")
        print("Instala con: pip install " + " ".join(missing))
        return
    
    # Iniciar sistema
    core = BETTERME_Core()
    BetterMeApp(core)


if __name__ == "__main__":
    main()
