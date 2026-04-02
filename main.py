import math
import time

from core.vision import VisionEngine
from core.physics import PhysicsBrain
from drivers import stealth

# Configuración del área de la ruleta (Ajustar a tu pantalla)
vision = VisionEngine({'top': 200, 'left': 500, 'width': 600, 'height': 600})
brain = PhysicsBrain()

print("ALEXBOT ACTIVADO - CALIBRANDO...")

while True:
    ball, zero, frame = vision.get_positions()

    if ball and zero:
        # 1. Calcular ángulos y velocidades
        omega = brain.estimate_omega(math.atan2(ball[1], ball[0]))

        # 2. Si la velocidad es la ideal para predecir (Ventana de apuesta)
        if 5.0 > omega > 2.0:
            prediction = brain.predict_drop_zone(omega, 1.2)
            print(f"Predicción: Zona {prediction} - ENVIANDO APUESTA")

            # 3. Ejecutar apuesta
            stealth.place_bet_safe(800, 400)  # Coordenadas del tapete
            time.sleep(15)  # Esperar a que termine la ronda
