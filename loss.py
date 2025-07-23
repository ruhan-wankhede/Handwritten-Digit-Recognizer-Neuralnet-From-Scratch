import numpy as np

class Loss:
    @staticmethod
    def categorical_cross_entropy(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def mean_squared_error(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (inputs - targets)**2