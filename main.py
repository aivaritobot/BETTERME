import json
from engine.vision import AlexBotVision
from engine.physics import AlexBotPhysics
from ui.overlay import show_alex_overlay
from utils.mapping import get_relative_prediction_angle


try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print('ALEXBOT ERROR: Primero debes ejecutar calibrate.py')
    raise SystemExit(1)

vision = AlexBotVision(config['roi'])
brain = AlexBotPhysics()

print('>>> ALEXBOT V3 PRO - SISTEMA DE PREDICCIÓN ACTIVADO')

while True:
    ball_angle, zero_angle, frame, debug = vision.get_alex_data()
    brain.update(ball_angle, zero_angle)
    telemetry = brain.get_prediction()

    relative_prediction = None
    if telemetry is not None:
        relative_prediction = get_relative_prediction_angle(
            telemetry.get('ball_pred'),
            telemetry.get('rotor_pred'),
        )

    show_alex_overlay(
        frame,
        relative_prediction=relative_prediction,
        telemetry=telemetry,
        debug=debug,
    )

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("ALEXBOT ERROR: Primero debes ejecutar calibrate.py")
    exit()

vision = AlexBotVision(config['roi'])
brain = AlexBotPhysics()

print(">>> ALEXBOT V3 PRO - SISTEMA DE PREDICCIÓN ACTIVADO")

while True:
    angle, frame = vision.get_alex_data()
    prediction = brain.get_prediction(angle) if angle else None
    show_alex_overlay(frame, prediction)
