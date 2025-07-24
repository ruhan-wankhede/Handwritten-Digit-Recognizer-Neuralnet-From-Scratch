import numpy as np

class ActivationFunctions:
    @staticmethod
    def ReLU(inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)

    @staticmethod
    def ReLU_derivative(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs > 0).astype(float)

    @staticmethod
    def softmax(inputs: np.ndarray) -> np.ndarray:
        exp = np.exp(inputs)
        return exp / np.sum(exp)