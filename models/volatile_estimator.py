import numpy as np

class VolatileEstimator:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.previous_aqi = None

    def detect_change(self, current_aqi: float) -> bool:
        if self.previous_aqi is None:
            self.previous_aqi = current_aqi
            return False

        change = abs(current_aqi - self.previous_aqi)
        self.previous_aqi = current_aqi
        return change >= self.threshold

# Example usage:
# estimator = VolatileEstimator(threshold=10)
# if estimator.detect_change(current_aqi):
#     print('Significant AQI change detected!')