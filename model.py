from typing import Callable
from tqdm import tqdm
import numpy as np
from activation import ActivationFunctions
from loss import Loss
import matplotlib.pyplot as plt
import json

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation: Callable[[np.ndarray], np.ndarray],
                 activation_derivative: Callable[[np.ndarray], np.ndarray] | None):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, inputs: np.ndarray) -> None:
        self.input = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation(self.z)

    def backprop(self, dC: np.ndarray) -> None:
        # derivative of activation function
        if self.activation_derivative is not None:
            dsigma = dC * self.activation_derivative(self.z)
        else:
            dsigma = dC

        # gradients
        self.dweights = np.dot(self.input.T, dsigma)
        self.dbiases = np.sum(dsigma, axis=0, keepdims=True)
        self.dinput = np.dot(dsigma, self.weights.T)


class StochasticGradientDescent:
    def __init__(self, alpha: float):
        """

        :param alpha: learning rate
        """
        self.alpha = alpha

    def update(self, layer):
        layer.weights -= self.alpha * layer.dweights
        layer.biases -= self.alpha * layer.dbiases


class Network:
    def __init__(self, layers: list[Layer], optimizer: StochasticGradientDescent):
        self.layers = layers
        self.optimizer = optimizer

    def save_json(self, path: str):
        model_data = []
        for layer in self.layers:
            model_data.append({
                "weights": layer.weights.tolist(),
                "biases": layer.biases.tolist(),
                "n_inputs": layer.weights.shape[0],
                "n_neurons": layer.weights.shape[1],
                "activation": layer.activation.__name__,
                "activation_derivative": (layer.activation_derivative.__name__ if layer.activation_derivative else None)
                # else implemented for Softmax + CCE where activation_derivative is None
            })
        with open(path, "w") as f:
            json.dump(model_data, f)

    def load_data(self) -> None:
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical

        # Load MNIST data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        # Flatten images to single dimensional array of 784 values from 28x28
        # / 255 to normalize pixel values between [0, 1]
        self.x_train = self.x_train.reshape(-1, 28 * 28) / 255
        self.x_test = self.x_test.reshape(-1, 28 * 28) / 255

        # one hot encoding labels
        self.y_train = to_categorical(self.y_train, num_classes=10)
        self.y_test = to_categorical(self.y_test, num_classes=10)

    def train(self, epochs: int, batch_size: int) -> None:
        self.loss_history = []
        for epoch in range(epochs):
            permutation = np.random.permutation(self.x_train.shape[0])
            x_shuffled = self.x_train[permutation]
            y_shuffled = self.y_train[permutation]

            epoch_loss = 0
            num_batches = self.x_train.shape[0] // batch_size


            for i in tqdm(range(0, self.x_train.shape[0], batch_size), total=num_batches, desc=f"Epoch {epoch + 1}", leave=True):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                a = x_batch # input to the current layer

                # feed forward
                for layer in self.layers:
                    layer.forward(a)
                    a = layer.output

                # calculate loss
                loss = Loss.categorical_cross_entropy(a, y_batch)
                epoch_loss += loss

                # Derivative of Loss, for Softmax + CCE; simply: prediction - target
                dloss = Loss.categorical_cross_entropy_derivative(a, y_batch)

                # backprop
                for layer in reversed(self.layers):
                    layer.backprop(dloss)
                    dloss = layer.dinput

                # Update weights and biases
                for layer in self.layers:
                    self.optimizer.update(layer)

            avg_loss = epoch_loss / num_batches
            tqdm.write(f"-> Loss: {avg_loss:.4f}")
            self.loss_history.append(avg_loss)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predicts the digit from x
        """
        # Make sure input is 2D (batch of 1 if single sample)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        a = x

        for layer in self.layers:
            layer.forward(a)
            a = layer.output

        return np.argmax(a, axis=1)

    def evaluate(self) -> None:
        """
        Evaluates accuracy of network on test data
        """
        predictions = self.predict(self.x_test)
        y_true = np.argmax(self.y_test, axis=1)
        accuracy = np.mean(predictions == y_true)
        print(f"Test Accuracy: {accuracy.astype(float) * 100:.2f}%")

    def plot_loss(self) -> None:
        """
        plot a graph depicting the loss as the model trains
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()


def train_model(path: str):
    """
    train the model and save the model data in the specified path
    """
    # Initializing the layers of the network
    layer1 = Layer(
        n_inputs=784,
        n_neurons=128,
        activation=ActivationFunctions.ReLU,
        activation_derivative=ActivationFunctions.ReLU_derivative,
    )

    layer2 = Layer(
        n_inputs=128,
        n_neurons=64,
        activation=ActivationFunctions.ReLU,
        activation_derivative=ActivationFunctions.ReLU_derivative,
    )

    layer3 = Layer(
        n_inputs=64,
        n_neurons=10,
        activation=ActivationFunctions.softmax,
        activation_derivative=None,  # Not needed for softmax + CCE
    )

    optimizer = StochasticGradientDescent(alpha=0.005)
    net = Network([layer1, layer2, layer3], optimizer)
    net.load_data()
    net.train(epochs=40, batch_size=32)
    net.save_json(path)


def load_model(path: str) -> Network:
    """
    load the model data from the JSON file
    """
    with open(path, "r") as f:
        model_data = json.load(f)

    layers = []
    
    for layer_info in model_data:

        # getting function references from the json text
        activation_name = layer_info["activation"]
        activation = getattr(ActivationFunctions, activation_name)

        deriv_name = layer_info["activation_derivative"]
        activation_derivative = getattr(ActivationFunctions, deriv_name) if deriv_name else None

        layer = Layer(
            n_inputs=layer_info["n_inputs"],
            n_neurons=layer_info["n_neurons"],
            activation=activation,
            activation_derivative=activation_derivative,
        )

        layer.weights = layer_info["weights"]
        layer.biases = layer_info["biases"]
        layers.append(layer)

    optimizer = StochasticGradientDescent(alpha=0.005)
    return Network(layers, optimizer)


    


