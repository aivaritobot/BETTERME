import json
from engine.vision import AlexBotVision
from engine.physics import AlexBotPhysics
from ui.overlay import show_alex_overlay

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
