import numpy as np


class FC:
    def __init__(self, input_size: int, output_size: int, name: str, initialize_method: str = "random"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.initialize_method = initialize_method
        self.parameters = [self.initialize_weights(), self.initialize_bias()]
        self.input_shape = None
        self.reshaped_shape = None

    def initialize_weights(self):
        np.random.seed(3)
        if self.initialize_method == "random":
            w = np.random.randn(self.output_size, self.input_size)
            return w * 0.01

        elif self.initialize_method == "xavier":
            return None

        elif self.initialize_method == "he":
            return None

        else:
            raise ValueError("Invalid initialization method")

    def initialize_bias(self):
        return np.zeros((self.output_size, 1))

    def forward(self, A_prev):
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)
        if A_prev.shape[0] != self.input_size:
            batch_size = A_prev.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T
        self.reshaped_shape = A_prev_tmp.shape
        W, b = self.parameters
        Z = np.dot(W, A_prev_tmp) + b
        return Z

    def backward(self, dZ, A_prev):
        A_prev_tmp = np.copy(A_prev)
        if A_prev.shape[0] != self.input_size:
            batch_size = A_prev.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T
            self.input_shape = A_prev_tmp.shape
        else:
            batch_size = dZ.shape[1]

        
        W, b = self.parameters
        dW = (1 / batch_size) * np.dot(dZ, A_prev_tmp.T)
        db = (1 / batch_size) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        grads = [dW, db]
        if A_prev.shape[0] != self.input_size:
            dA_prev = dA_prev.T.reshape(self.input_shape)
        return dA_prev, grads

    def update_parameters(self, optimizer, grads):
        self.parameters = optimizer.update(grads, self.name)
