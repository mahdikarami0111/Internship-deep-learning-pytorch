from layers.convolution2d import Conv2D
from layers.maxpooling2d import MaxPool2D
from layers.fullyconnected import FC

from activations import Activation, get_activation

import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


class Model:
    def __init__(self, arch, criterion, optimizer, name=None):
        if name is None:
            self.model = arch
            self.criterion = criterion
            self.optimizer = optimizer
            self.layers_names = list(arch.keys())
        else:
            self.model, self.criterion, self.optimizer, self.layers_names = self.load_model(name)

    def is_layer(self, layer):

        return isinstance(layer, FC) or isinstance(layer, Conv2D) or isinstance(layer, MaxPool2D)

    def is_activation(self, layer):
        return isinstance(layer, Activation)

    def forward(self, x):
        tmp = []
        A = x
        for l in range(len(self.layers_names)):
            layer = self.model[self.layers_names[l]]
            if self.is_layer(layer):
                Z = layer.forward(A)
                tmp.append(Z)
            elif self.is_activation(layer):
                A = layer.forward(Z)
                tmp.append(A)
        return tmp

    def backward(self, dAL, tmp, x):
        dA = dAL
        grads = {}
        for l in reversed(range(1, len(self.layers_names), 2)):
            if l > 2:
                Z, A = tmp[l - 1], tmp[l - 2]
            else:
                Z, A = tmp[l - 1], x
            activation = self.model[self.layers_names[l]]
            layer = self.model[self.layers_names[l-1]]
            dZ = activation.backward(dA, Z)
            dA, grad = layer.backward(dZ, A)
            grads[self.layers_names[l - 1]] = grad

        return grads

    def update(self, grads):
        i = 0
        for l in range(len(self.layers_names)):
            layer = self.model[self.layers_names[l]]
            if self.is_layer(layer) and not isinstance(layer, MaxPool2D):
                layer.update_parameters(self.optimizer, grads[self.layers_names[l]])

    def one_epoch(self, x, y):
        tmp = self.forward(x)
        AL = tmp[-1]
        loss = self.criterion.compute(AL, y)

        dAL = self.criterion.backward(AL, y)
        grads = self.backward(dAL, tmp, x)
        self.update(grads)
        return loss

    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump((self.model, self.criterion, self.optimizer, self.layers_names), f)

    def load_model(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def shuffle(self, m, shuffling):
        order = list(range(m))

        if shuffling:
            np.random.shuffle(order)
            return order
        return order

    def batch(self, X, y, batch_size, index, order):
        last_index = ((index + 1) * batch_size) - 1  # hint last index of the batch check for the last batch
        batch = order[index * batch_size: last_index]
        if X.ndim == 4:
            bx = X[batch, :, :, :]
            by = y[batch]
            return bx, by
        else:
            bx = X[:, batch]
            by = y[:, batch]
            return bx, by

    def compute_loss(self, X, y):
        tmp = self.forward(X)
        AL = tmp[-1]
        return self.criterion.compute(AL, y)

    def train(self, X, y, X_test, Y_test, epochs, val=None, batch_size=3, shuffling=False, verbose=1, save_after=None):
        train_cost = []
        val_cost = []
        if X.ndim == 4:
            m = X.shape[0]
        elif X.ndim == 2:
            m = X.shape[1]
        else:
            raise Exception("Invalid input dimension")

        for e in range(1, epochs + 1):
            order = self.shuffle(m, shuffling)
            cost = 0
            n = m // batch_size
            for b in range(m // batch_size):
                bx, by = self.batch(X, y, batch_size, b, order)
                cost += self.one_epoch(bx, by)
            train_cost.append(cost/n)
            if val is not None:
                val_cost.append(self.compute_loss(val[0], val[y], batch_size))
            if verbose != False:
                prediction = np.floor(2 * self.predict(X_test))
                accuracy = 100 * (1 - np.sum(np.absolute(prediction - Y_test)) / Y_test.shape[1])
                if e % verbose == 0:
                    print(accuracy)
                    print("Epoch {}: train cost = {}".format(e, cost))
                if val is not None:
                    print("Epoch {}: val cost = {}".format(e, val_cost[-1]))
        if save_after is not None:
            self.save(save_after)
        return train_cost, val_cost

    def predict(self, X):
        return self.forward(X)[-1]

