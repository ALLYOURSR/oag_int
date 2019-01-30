import numpy as np
from scipy import stats

class ErrorTracker:
    # Cyclical container for continuously averaging a window of error in a fixed size data structure
    def __init__(self, array_length):
        self.array_length = array_length
        self.errors = np.full(self.array_length, 99999999999.0)
        self.last_index = 0

    def add_error(self, value):
        self.errors[self.last_index] = value
        self.last_index += 1
        self.last_index = self.last_index % self.array_length

    def get_current_average(self):
        return np.mean(self.errors[:self.last_index])

    def get_slope(self):
        # Calculates a linear regression of the current errors to see if the net is still learning. Use as quit condition

        #Rotate array to preserve time series order
        a0 = self.errors[self.last_index:]
        a1 = self.errors[:self.last_index]
        y = np.concatenate([a0, a1], 0)

        x = np.array(range(len(y)))
        s, _, _, _, _ = stats.linregress(x, y)

        return s
