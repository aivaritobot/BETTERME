import time
import math
import pyautogui
import random
from vision import RouletteEye

# --- CONFIGURACIÓN DE TU PANTALLA ---
# Aquí pondrás los números de donde está el video de la ruleta
X, Y, W, H = 500, 200, 400, 400  
CENTRO = (200, 200)             
FRICCION = 0.04                 

def move_mouse(x_dest, y_dest):
    # Mueve el ratón de forma curva para que no parezca un bot
    pyautogui.moveTo(x_dest, y_dest, duration=random.uniform(0.5, 1.0), tween=pyautogui.easeInOutQuad)

# Iniciamos el ojo con las coordenadas configuradas
eye = RouletteEye(X, Y, W, H)
history = []

print(">>> BOT alAIve INICIADO. Analizando trayectoria...")

while True:
    ball = eye.get_ball()
    if ball:
        history.append(ball)
        if len(history) > 20:
            # Cálculo de la trayectoria
            dx, dy = ball[0] - CENTRO[0], ball[1] - CENTRO[1]
            angle = math.atan2(dy, dx)
            print(f"BOLA DETECTADA: Predicción zona de caída... Ángulo: {math.degrees(angle):.0f}°")
    else:
        history = []
    time.sleep(0.01)
