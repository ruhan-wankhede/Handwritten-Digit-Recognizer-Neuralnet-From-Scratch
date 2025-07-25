import numpy as np

class ActivationFunctions:
    @staticmethod
    def ReLU(inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)

    @staticmethod
    def ReLU_derivative(inputs: np.ndarray) -> np.ndarray:
        return (inputs > 0).astype(float)

    @staticmethod
    def softmax(inputs: np.ndarray) -> np.ndarray:
        # subtract max per row for numerical stability
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp = np.exp(shifted_inputs)
        return exp / np.sum(exp, axis=1, keepdims=True)