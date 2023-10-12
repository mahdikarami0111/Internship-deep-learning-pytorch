import numpy as np

class MeanSquaredError:
    def __init__(self):
        pass

    def compute(self, y_pred, y_true):
        batch_size = y_true.shape[1]
        cost = (1/batch_size) * np.square(y_pred - y_true)
        cost = np.sum(cost)
        return np.squeeze(cost)
    
    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true)