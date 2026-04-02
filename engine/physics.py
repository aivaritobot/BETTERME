import time

class AlexBotPhysics:
    def __init__(self):
        self.history = [] 
        self.friction_alpha = 0.024 # Ajustable según mesa

    def get_prediction(self, angle):
        t = time.time()
        self.history.append((t, angle))
        if len(self.history) > 12: self.history.pop(0)
        if len(self.history) < 4: return None

        # Velocidad Angular ALEXBOT
        dt = self.history[-1][0] - self.history[-2][0]
        if dt == 0: return None
        
        d_theta = (self.history[-1][1] - self.history[-2][1]) % 360
        omega = d_theta / dt

        # Predicción Cinética ALEXBOT
        if 25 > omega > 4:
            pred = (self.history[-1][1] + (omega**2 / (2 * self.friction_alpha))) % 360
            return pred
        return None
