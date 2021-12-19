import numpy as np


class CategoricalCrossEntropy:
    def __call__(self, target, output):
        return -np.sum(target * np.log(output))

    def derivative(self, target, output):
        return output - target
