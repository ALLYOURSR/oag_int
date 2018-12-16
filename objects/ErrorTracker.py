import numpy as np

class ErrorTracker:
    # Cyclical container for continuously averaging a window of error in a fixed size data structure
    def __init__(self, array_length):
        self.array_length = array_length
        self.errors = np.zeros(self.array_length)
        self.last_index = 0

    def add_error(self, value):
        if len(self.errors) < self.array_length:
            self.errors[self.last_index] = value
            self.last_index += 1
        else:
            add_index = self.last_index % self.array_length
            self.errors[add_index] = value
            self.last_index += 1

    def get_current_average(self):
        return np.mean(self.errors)