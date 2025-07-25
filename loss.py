import numpy as np

class Loss:
    @staticmethod
    def categorical_cross_entropy(predictions: np.ndarray, targets: np.ndarray):
        # avoiding log(0)
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)

        # confidence of the model of the target digits
        correct_confidences = np.sum(predictions * targets, axis=1)

        return -np.mean(np.log(correct_confidences))

    @staticmethod
    def categorical_cross_entropy_derivative(predictions: np.ndarray, targets: np.ndarray):
        # Simply predictions - targets for Softmax + CCE
        return (predictions - targets) / predictions.shape[0]

    @staticmethod
    def mean_squared_error(predictions: np.ndarray, targets: np.ndarray):
        return (predictions - targets)**2