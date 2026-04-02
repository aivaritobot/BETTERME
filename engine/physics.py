import time


class AlexBotPhysics:
    def __init__(self):
        self.ball_history = []
        self.rotor_history = []
        self.max_history = 24
        self.drop_omega_threshold = 3.2

    def update(self, ball_angle, rotor_angle=None):
        now = time.time()
        if ball_angle is not None:
            self.ball_history.append((now, ball_angle))
        if rotor_angle is not None:
            self.rotor_history.append((now, rotor_angle))
        self._trim()

    def _trim(self):
        if len(self.ball_history) > self.max_history:
            self.ball_history = self.ball_history[-self.max_history:]
        if len(self.rotor_history) > self.max_history:
            self.rotor_history = self.rotor_history[-self.max_history:]

    @staticmethod
    def _angle_delta(a1, a2):
        return ((a2 - a1 + 180.0) % 360.0) - 180.0

    @classmethod
    def _estimate_kinematics(cls, history):
        if len(history) < 5:
            return None, None

        # velocidades instantáneas
        omegas = []
        omega_times = []
        for i in range(1, len(history)):
            t0, a0 = history[i - 1]
            t1, a1 = history[i]
            dt = t1 - t0
            if dt <= 1e-5:
                continue
            dtheta = cls._angle_delta(a0, a1)
            omegas.append(dtheta / dt)
            omega_times.append((t0 + t1) * 0.5)

        if len(omegas) < 4:
            return None, None

        omega = sum(omegas[-4:]) / min(4, len(omegas))

        # desaceleración (alpha) estimada de forma local
        alpha_samples = []
        for i in range(1, len(omegas)):
            dt = omega_times[i] - omega_times[i - 1]
            if dt <= 1e-5:
                continue
            alpha_samples.append((omegas[i] - omegas[i - 1]) / dt)

        if len(alpha_samples) < 2:
            return omega, None

        alpha = sum(alpha_samples[-4:]) / min(4, len(alpha_samples))
        return omega, alpha

    def get_prediction(self):
        ball_omega, ball_alpha = self._estimate_kinematics(self.ball_history)
        if ball_omega is None or ball_alpha is None:
            return None

        # Buscamos desaceleración real (negativa para giro antihorario/horario según convención)
        if abs(ball_omega) < self.drop_omega_threshold or abs(ball_alpha) < 1e-3:
            return None

        # tiempo estimado a umbral de caída
        target = self.drop_omega_threshold if ball_omega > 0 else -self.drop_omega_threshold
        t_drop = (target - ball_omega) / ball_alpha
        if not (0.2 <= t_drop <= 6.0):
            return None

        ball_now = self.ball_history[-1][1]
        ball_pred = (ball_now + ball_omega * t_drop + 0.5 * ball_alpha * (t_drop ** 2)) % 360

        rotor_omega, rotor_alpha = self._estimate_kinematics(self.rotor_history)
        rotor_pred = None
        if rotor_omega is not None:
            rotor_alpha = rotor_alpha if rotor_alpha is not None else 0.0
            rotor_now = self.rotor_history[-1][1]
            rotor_pred = (rotor_now + rotor_omega * t_drop + 0.5 * rotor_alpha * (t_drop ** 2)) % 360

        return {
            'ball_pred': ball_pred,
            'rotor_pred': rotor_pred,
            't_drop': t_drop,
            'ball_omega': ball_omega,
            'ball_alpha': ball_alpha,
        }
