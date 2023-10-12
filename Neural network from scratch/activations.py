import numpy as np
from abc import ABC, abstractmethod

class Activation:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        pass

class Sigmoid(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = 1 / (1 + np.exp(-Z))
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        A = 1 / (1 + np.exp(-Z))
        dZ = dA * A * (1-A)
        return dZ
    

class ReLU(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = np.maximum(0, Z)
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ
    
    

class Tanh(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = (np.exp(2 * Z) - 1)/(np.exp(2 * Z) + 1)
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        A = self.forward(Z)
        dZ = dA * (1 - (np.multiply(A, A)))
        return dZ
    
class LinearActivation(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = Z
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dZ = dA
        return dZ

def get_activation(activation: str) -> tuple:
    if activation == 'sigmoid':
        return Sigmoid
    elif activation == 'relu':
        return ReLU
    elif activation == 'tanh':
        return Tanh
    elif activation == 'linear':
        return LinearActivation
    else:
        raise ValueError('Activation function not supported')