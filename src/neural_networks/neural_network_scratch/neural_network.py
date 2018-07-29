import numpy as np
from math import e

'''
Author: Shyam Padia
Date: 07/28/2018
An Algorithm that implements a simple Multi Perceptron Neural Network using only Numpy
This algorithm also implements Gradient Descent and Backpropagation
'''


class Layer:
    # (previous layers size, current layer size)
    def __init__(self, size, activation=None, prev_layer=None, next_layer=None):
        self.size = size
        self.activation = activation
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.neurons = np.zeros((size, 1))
        if self.prev_layer is not None:
            self.biases = np.random.rand((size, 1))
            self.weights = np.random.rand((self.prev_layer.size, size))
            self.prev_layer.next_layer = self


def activate(vals, func):
    # vals format n x 1
    if func == 'relu':
        for v in vals:
            v[1] = max(0, v[1])
    elif func == 'sigmoid':
        for v in vals:
            v[1] = 1 / (1 + e**-v[1])
    return vals

class NeuralNetwork:
    def __init__(self, input_size=None):
        self.input_layer = Layer(input_size)
        self.last_layer = None

    def fully_connected(self, incoming_layer, size, activation='relu'):
        self.last_layer = Layer(size, activation=activation, prev_layer=incoming_layer)

    def fit(self, inputs, target, epochs=1):
        for epoch in range(epochs):
            # inp expected as (input_size x 1) numpy array
            # do cross validation
            for inp in inputs:
                self.input_layer.neurons = inp
                self.forward_prop()
                self.back_prop(target)

    # can also be used for predict
    def forward_prop(self):
        curr_layer = self.input_layer
        while curr_layer is not None:
            curr_layer.next_layer.neurons = activate(
                np.matmul(curr_layer.next_layer.weights, curr_layer.neurons) + curr_layer.next_layer.biases,
                curr_layer.activation)
            curr_layer = curr_layer.next_layer

    # def back_prop(self, target):
