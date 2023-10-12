import numpy as np

class BinaryCrossEntropy:
    def __init__(self) -> None:
        pass

    def compute(self, Y_hat: np.ndarray, Y: np.ndarray) -> float:
        epsilon = 1e-9
        m = Y.shape[1]
        cost = (-1 / m) * (np.dot(Y, np.log(Y_hat + epsilon).T) + np.dot((1 - Y), np.log(1 - Y_hat + epsilon).T))
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        return cost

    def backward(self, Y_hat: np.ndarray, Y: np.ndarray) -> np.ndarray:
        epsilon = 1e-9
        return - (np.divide(Y, Y_hat + epsilon) - np.divide(1 - Y, 1 - (Y_hat + epsilon)))

