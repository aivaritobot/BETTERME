import pyautogui
import random
import time
import numpy as np

def move_human(x, y):
    # Genera puntos intermedios para una trayectoria curva
    start_x, start_y = pyautogui.position()
    cp1_x = start_x + random.randint(-100, 100)
    cp1_y = start_y + random.randint(-100, 100)
    
    # Simulación de aceleración humana (Ease-in-out)
    pyautogui.moveTo(x + random.uniform(-2, 2), y + random.uniform(-2, 2), 
                     duration=random.uniform(0.3, 0.6), 
                     tween=pyautogui.easeInOutQuad)

def place_bet_safe(x, y):
    move_human(x, y)
    time.sleep(random.uniform(0.05, 0.1))
    pyautogui.click()
