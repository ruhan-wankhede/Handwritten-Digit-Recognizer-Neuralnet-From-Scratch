import numpy as np

class ActivationFunctions:
    @staticmethod
    def ReLU(inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)

