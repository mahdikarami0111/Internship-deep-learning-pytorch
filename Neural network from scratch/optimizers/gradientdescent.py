# TODO: Implement the gradient descent optimizer
import numpy as np
class GD:
    def __init__(self, layers_list: dict, learning_rate: float):
        self.learning_rate = learning_rate
        self.layers = layers_list
        self.sign = 1


    def update(self, grads, name):
        layer = self.layers[name]
        params = []
        for i in range(len(grads)):
            params.append(layer.parameters[i] - grads[i] * self.learning_rate)
        return params