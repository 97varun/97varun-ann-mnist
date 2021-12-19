from sklearn.metrics import confusion_matrix
import sys
from time import time
import numpy as np
from numpy.core.fromnumeric import reshape, shape
from LossFunctions import CategoricalCrossEntropy
import NNConstants
from ActivationFunctions import Sigmoid, Softmax


class Neuron:
    """
    A class to represent a neuron in the in a layer in the artificial neural network.
    """

    def __init__(self, size):
        """
        Randomly initialize the neuron weights and bias.
        """
        self.weight = np.random.uniform(
            NNConstants.MIN_INIT_WEIGHT, NNConstants.MAX_INIT_WEIGHT, size=(size,))
        self.bias = np.random.uniform(
            NNConstants.MIN_INIT_WEIGHT, NNConstants.MAX_INIT_WEIGHT)

    def get_output(self, inputs):
        """
        Return the output of the neuron for the given inputs.
        """

        return np.dot(self.weight, inputs) + self.bias

    def update_weight(self, error, delta):
        """
        Update weights and biases
        """
        self.weight -= error
        self.bias -= delta


class Layer:
    """
    A layer in the artificial neural network.
    """

    def __init__(self,
                 num_inputs,
                 num_neurons,
                 is_output=False,
                 activation_function=Sigmoid()):
        """
        Initialize the layer with num_neurons neurons, each with num_inputs inputs, whether it's output and activation function.
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs

        self.neurons = [
            Neuron(num_inputs) for _ in range(num_neurons)]
        self.is_output = is_output

        self.activation_function = activation_function

        if is_output:
            self.activation_function = Softmax()

        self.loss_function = CategoricalCrossEntropy()

        self.reset_errors()
        self.reset_loss()

    def compute_activations(self, x):
        """
        Compute activations for this layer using outputs from previous layer.
        """
        self.inputs = x

        if self.is_output:
            self.activations = self.activation_function(
                np.fromiter(map(lambda neuron: neuron.get_output(x), self.neurons), dtype='float64'))
        else:
            self.activations = np.fromiter(map(lambda neuron: self.activation_function(
                neuron.get_output(x)), self.neurons), dtype='float64')

    def compute_errors(self, downstream_layer):
        """
        Compute deltas for updating weights using current layer's activations and weights, deltas from downstream layer.

        1) For hidden layers,

             |----------------first part---------------------|  |------------second part---------------|
        dj = derivative of activation function (substitute oj) * sigma k over downstream(j) [ dk * wkj ]

        2) For output layer,

        dj = derivative of loss function * derivative of activation function

        """

        if self.is_output:
            self.deltas = self.loss_function.derivative(
                downstream_layer, self.activations)

            self.loss += self.loss_function(downstream_layer, self.activations)

        else:
            self.deltas = []

            for j, activation in enumerate(self.activations):
                first_part = self.activation_function.derivative(activation)

                second_part = 0
                for downstream_layer_neuron, deltak in zip(downstream_layer.neurons, downstream_layer.deltas):
                    second_part += deltak * downstream_layer_neuron.weight[j]

                self.deltas.append(first_part * second_part)

            self.deltas = np.array(self.deltas)

        # dwji = dj * xji
        self.errors = np.matmul(
            np.reshape(self.deltas, newshape=(len(self.deltas), 1)),
            np.reshape(self.inputs, newshape=(1, len(self.inputs)))
        )

    def update_weights(self, learning_rate):
        """
        Update weights using deltas computed during backpropagation.
        """
        for neuron, err, delta in zip(self.neurons, self.errors, self.deltas):
            neuron.update_weight(learning_rate * err, learning_rate * delta)

    def reset_errors(self):
        self.errors = np.zeros(shape=(self.num_neurons, self.num_inputs))

    def reset_loss(self):
        self.loss = 0


class ArtificialNeuralNetwork:
    """
    Artificial Neural Network from scratch.
    """

    def __init__(self,
                 learning_rate=NNConstants.DEFAULT_LEARNING_RATE,
                 batch_size=NNConstants.DEFAULT_BATCH_SIZE):
        """
        Initialize the Artificial Neural Network.
        """
        self.layers = []
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    layer_count = 1

    def add_layer(self, layer):
        """
        Add a layer to the artificial neural network.
        """
        layer.name = f'layer_{ArtificialNeuralNetwork.layer_count}'
        ArtificialNeuralNetwork.layer_count += 1

        self.layers.append(layer)

    def forward_propagation(self, x):
        """
        Propogate input forward and calculate activations for each layer.
        """
        prev_activations = x
        for layer in self.layers:
            layer.compute_activations(prev_activations)
            prev_activations = layer.activations

    def backward_propagation(self, y):
        """
        Propogate backward and compute errors at each layer.
        """
        self.layers[-1].compute_errors(y)

        downstream_layer = self.layers[-1]

        # All layers except the last layer
        for layer in reversed(self.layers[:-1]):
            layer.compute_errors(downstream_layer)
            downstream_layer = layer

    def update_weights(self):
        """
        Update weights with the errors computer from backward_propogation.
        """
        for layer in self.layers:
            layer.update_weights(self.learning_rate)
            # layer.reset_errors()

    def train(self, x, y, val_x, val_y, epochs):
        """
        Train the network, taking self.batch_size samples at a time.
        """
        for epoch in range(epochs):
            start_time = time()

            for xi, yi in zip(x, y):
                self.forward_propagation(xi)
                self.backward_propagation(yi)

                self.update_weights()

            print(
                f'epoch: {epoch}, time: {time() - start_time}, loss: {self.layers[-1].loss}, val_accuracy: {self.validate(val_x, val_y)}', flush=True)

            self.layers[-1].reset_loss()

    def validate(self, x, y):
        network_output = self.predict(x)

        # Convert softmax predictions (probability distributions) to labels
        predictions = get_labels_from_network_output(network_output)

        accuracy = get_accuracy(y, predictions)
        return accuracy

    def predict(self, x):
        """
        Predict the softmax distribution given input x.
        """
        predictions = []

        for xi in x:
            self.forward_propagation(xi)
            predictions.append(self.layers[-1].activations)

        return np.array(predictions)


def get_random_predictions(num_predictions):
    """
    Returns a random array of predictions.
    """
    return np.random.randint(0, NNConstants.NUM_CLASSES, num_predictions)


def get_labels_from_network_output(network_output):
    return np.fromiter(map(lambda y: np.argmax(y), network_output), dtype='int64')


def one_hot_encode(labels):
    labels = labels.astype('int64')
    num_labels = labels.shape[0]

    one_hot_encoded_labels = np.zeros(
        shape=(num_labels, NNConstants.NUM_CLASSES))
    one_hot_encoded_labels[np.arange(num_labels), labels] = 1

    return one_hot_encoded_labels


def get_accuracy(labels, predictions):
    return 1 - np.count_nonzero(labels - predictions) / labels.shape[0]


def normalize(input, mn, mx, a, b):
    return a + ((input - mn) * (b - a)) / (mx - mn)


def augment(train_images, train_labels, digit):
    digits = train_labels == digit

    train_labels = np.append(train_labels, train_labels[digits])

    trueArr = np.ones(shape=(1, train_images.shape[1]), dtype=bool)
    digits = np.reshape(digits, newshape=(digits.shape[0], 1))
    digits = np.matmul(digits, trueArr)

    train_images = np.append(train_images, train_images[digits])

    train_images = np.reshape(train_images, newshape=(len(train_labels), 784))

    print(train_labels.shape, train_images.shape)

    return train_images, train_labels


if __name__ == "__main__":
    train_image_file, train_label_file, test_image_file = sys.argv[1:]

    train_images = np.genfromtxt(train_image_file, delimiter=',')
    train_labels = np.genfromtxt(train_label_file, delimiter=',')
    test_images = np.genfromtxt(test_image_file, delimiter=',')
    test_labels = np.genfromtxt('custom_test_label.csv', delimiter=',')

    random_indices = np.random.choice(
        train_images.shape[0], 10000, replace=False)
    train_images = train_images[random_indices]
    train_labels = train_labels[random_indices]

    train_images_processed = train_images / 255.0
    test_images_processed = test_images / 255.0

    # One-hot encode labels
    train_labels_processed = one_hot_encode(train_labels)

    ann = ArtificialNeuralNetwork()
    ann.add_layer(Layer(num_inputs=784, num_neurons=128,
                  activation_function=Sigmoid()))
    ann.add_layer(Layer(num_inputs=128, num_neurons=10, is_output=True))

    # Train the network
    ann.train(train_images_processed, train_labels_processed,
              test_images_processed, test_labels, epochs=10)
    network_output = ann.predict(test_images_processed)

    # Convert softmax predictions (probability distributions) to labels
    predictions = get_labels_from_network_output(network_output)
    print(predictions)

    print(f'confusion matrix:-\n{confusion_matrix(test_labels, predictions)}')
    accuracy = get_accuracy(test_labels, predictions)
    print(f'accuracy: {accuracy}')

    # Write predictions to file
    np.savetxt(NNConstants.TEST_PREDICTIONS_FILE,
               predictions, delimiter=',', fmt='%d')
