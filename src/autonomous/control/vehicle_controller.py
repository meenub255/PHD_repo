"""Low-level vehicle actuator controller managing steering angle, velocity, and braking state."""


class VehicleController:
    def __init__(self, mode='simulation'):
        self.mode = mode
        self.steering = 0.0
        self.speed = 0.0
        self.engine_on = True
        self.brake_active = False

    def apply_control(self, steering_angle, speed):
        self.steering = float(steering_angle)
        self.speed = float(speed)
        self.brake_active = self.speed < 5

    def emergency_brake(self):
        self.speed = 0.0
        self.steering = 0.0
        self.brake_active = True

    def get_status(self):
        return {
            'steering_angle': round(self.steering, 1),
            'speed': round(self.speed, 1),
            'engine_on': self.engine_on,
            'brake_active': self.brake_active,
            'mode': self.mode
        }