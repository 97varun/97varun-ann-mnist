import numpy as np


class ActivationFunction:
    def __call__():
        pass

    def derivative():
        pass


class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class Tanh(ActivationFunction):
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2


class Softmax(ActivationFunction):
    def __call__(self, x):
        exponent_vector = np.exp(x)
        def softmax(xi): return np.exp(xi) / np.sum(exponent_vector)
        return np.vectorize(softmax)(x)


class Identity(ActivationFunction):
    def __call__(self, x):
        return x
