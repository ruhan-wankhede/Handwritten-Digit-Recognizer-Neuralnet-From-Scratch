import numpy as np

class Loss:
    @staticmethod
    def categorical_cross_entropy(inputs: np.ndarray, targets: np.ndarray):
        # avoiding log(0)
        inputs = np.clip(inputs, 1e-9, 1 - 1e-9)

        # confidence of the model of the target digits
        correct_confidences = np.sum(inputs * targets, axis=1)

        return -np.mean(np.log(correct_confidences))

    @staticmethod
    def mean_squared_error(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (inputs - targets)**2