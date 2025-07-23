from typing import Callable

import numpy as np
from activation import ActivationFunctions

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation: Callable[[np.ndarray], np.ndarray]):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.output = None

    def forward(self, inputs: np.ndarray) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases

    def activate(self, inputs: np.ndarray) -> None:
        self.output = self.activation(self.output)

    def calculate_loss(self):
        pass

