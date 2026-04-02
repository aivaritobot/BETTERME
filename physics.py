import time
import math

class PhysicsBrain:
    def __init__(self):
        self.ball_history = [] # Almacena (tiempo, ángulo)
        self.friction_alpha = 0.005 # Coeficiente dinámico
        
    def estimate_omega(self, current_angle):
        t = time.time()
        self.ball_history.append((t, current_angle))
        if len(self.ball_history) < 2: return 0
        
        # Calcular velocidad angular (Omega)
        dt = self.ball_history[-1][0] - self.ball_history[-2][0]
        d_theta = self.ball_history[-1][1] - self.ball_history[-2][1]
        return d_theta / dt

    def predict_drop_zone(self, ball_omega, rotor_omega):
        # Punto crítico: cuando la bola baja de X rad/s, cae.
        # T_to_drop = (ball_omega - threshold_omega) / self.friction_alpha
        # Predecimos la posición del rotor en T_to_drop
        prediction = (ball_omega * 1.5) + rotor_omega # Simplificación vectorial
        return prediction % 360
