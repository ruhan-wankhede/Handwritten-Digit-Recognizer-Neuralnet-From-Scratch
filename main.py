from typing import Callable

import numpy as np
from activation import ActivationFunctions
from loss import Loss

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation: Callable[[np.ndarray], np.ndarray],
                 activation_derivative: Callable[[np.ndarray], np.ndarray], loss: Callable[[np.ndarray], np.ndarray]):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.loss_fn = loss

    def forward(self, inputs: np.ndarray) -> None:
        self.input = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation(self.z)

    def backprop(self, dC: np.ndarray) -> np.ndarray:
        # derivative of activation function
        dsigma = dC * self.activation_derivative(self.z)

        # gradients
        self.dweights = np.dot(self.input.T, dsigma)
        self.dbiases = np.sum(dsigma, axis=0)
        self.dinput = np.dot(dsigma, self.weights.T)



