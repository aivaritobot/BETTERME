from __future__ import annotations

# Orden oficial del cilindro Europeo
WHEEL_ORDER = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]

SECTORS = {
    'VECINOS DEL CERO': {'nums': [22, 18, 29, 7, 28, 12, 35, 3, 26, 0, 32, 15, 19, 4, 21, 2, 25], 'range': (305, 55)},
    'ORFALINOS': {'nums': [1, 20, 14, 31, 9, 17, 34, 6], 'range': (56, 110)},
    'SERIE 5/8 (TIER)': {'nums': [27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33], 'range': (111, 235)},
    'ORFALINOS_B': {'nums': [17, 34, 6], 'range': (236, 304)},
}


def get_alexbot_sector(angle: float):
    angle = angle % 360
    for name, data in SECTORS.items():
        low, high = data['range']
        if low < high:
            if low <= angle <= high:
                return name, data['nums']
        else:
            if angle >= low or angle <= high:
                return name, data['nums']
    return 'ALEXBOT ANALIZANDO...', []


def get_relative_prediction_angle(ball_pred: float | None, rotor_pred: float | None):
    if ball_pred is None:
        return None
    if rotor_pred is None:
        return ball_pred % 360
    return (ball_pred - rotor_pred) % 360
