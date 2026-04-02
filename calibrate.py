import cv2
import numpy as np
import mss
import json

def auto_calibrate():
    sct = mss.mss()
    # Capturamos la pantalla completa para buscar la ruleta
    monitor = sct.monitors[1]
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    frame = cv2.medianBlur(frame, 5)

    print("Buscando la ruleta en pantalla...")
    
    # Detectar círculos (Ruleta)
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=30, minRadius=100, maxRadius=400)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = circles[0, 0] # Tomamos el círculo más prominente
        
        config = {
            "center_x": int(best_circle[0]),
            "center_y": int(best_circle[1]),
            "radius": int(best_circle[2]),
            "roi": {
                "top": int(best_circle[1] - best_circle[2]),
                "left": int(best_circle[0] - best_circle[2]),
                "width": int(best_circle[2] * 2),
                "height": int(best_circle[2] * 2)
            }
        }
        
        with open('config.json', 'w') as f:
            json.dump(config, f)
        
        print(f"Calibración exitosa: Centro en {config['center_x']},{config['center_y']}")
        return config
    else:
        print("No se encontró la ruleta. Asegúrate de que el juego esté visible.")
        return None

if __name__ == "__main__":
    auto_calibrate()
