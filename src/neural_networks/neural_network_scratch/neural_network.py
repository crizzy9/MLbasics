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


class NeuralNetwork:
    def __init__(self, input_size=None):
        self.input_layer = Layer(input_size)
        self.last_layer = None

    def fully_connected(self, incoming_layer, size, activation='relu'):
        self.last_layer = Layer(size, activation=activation, prev_layer=incoming_layer)

    def fit(self, inputs, targets, epochs=1):
        for epoch in range(epochs):
            # inp expected as (input_size x 1) numpy array
            # do cross validation
            self.stochastic_gradient_descent(inputs, targets)

    @staticmethod
    def activate(vals, func):
        # vals format n x 1
        # also do `softmax`
        if func == 'relu':
            for v in vals:
                v[0] = max(0, v[0])
        elif func == 'sigmoid':
            for v in vals:
                v[0] = 1 / (1 + e ** -v[0])
        return vals

    # can also be used for predict
    def feed_forward(self):
        curr_layer = self.input_layer
        while curr_layer is not None:
            curr_layer.next_layer.neurons = NeuralNetwork.activate(
                np.dot(curr_layer.next_layer.weights, curr_layer.neurons) + curr_layer.next_layer.biases,
                curr_layer.activation)
            curr_layer = curr_layer.next_layer

    def stochastic_gradient_descent(self, inputs, targets, step_size=50):
        # divide input into stochastic steps
        steps = [(i, i+step_size) for i in range(0, len(inputs), step_size)]
        for start, end in steps:
            self.update_step(inputs[start:end], targets[start:end])


    def update_step(self, inputs, targets):
        for i in range(len(inputs)):
            self.input_layer.neurons = inputs[i]
            self.feed_forward()
            self.back_prop(targets[i])

    @staticmethod
    def rmse(out, target):
        # out and target n x 1
        return (out - target)**2

    def back_prop(self, target):
        error = NeuralNetwork.rmse(self.last_layer.neurons, target)
        # delta_weights =
