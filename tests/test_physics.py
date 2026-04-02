from engine.physics import AlexBotPhysics


def test_angle_delta_wraps():
    assert AlexBotPhysics._angle_delta(359, 1) == 2
    assert AlexBotPhysics._angle_delta(1, 359) == -2


def test_prediction_object_generation():
    brain = AlexBotPhysics()

    base_t = 1000.0
    ball = [40, 49, 57, 64, 70, 75, 79, 82, 84]
    rotor = [200, 202, 204, 206, 208, 210, 212, 214, 216]

    brain.ball_history = [(base_t + i * 0.2, a) for i, a in enumerate(ball)]
    brain.rotor_history = [(base_t + i * 0.2, a) for i, a in enumerate(rotor)]

    pred = brain.get_prediction()
    assert pred is not None
    assert 0 <= pred.ball_pred < 360
    assert 0 <= pred.impact_angle < 360
    assert 0.0 <= pred.confidence <= 1.0
