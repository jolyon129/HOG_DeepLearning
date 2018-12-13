import numpy as np
import sys
import os
import glob
import time
import csv


class NeuralNetwork:
    def __init__(self):
        np.random.seed(2)
        self.training_set_inputs = None
        self.y = None
        self.hidden_layer = None
        self.output = None
        self.learning_rate = None
        self.weights1 = None
        self.weights2 = None
        self.size_of_hidden_layer = None

    def build_network(self, x, y, size_of_hidden_layer=1):
        self.y = y
        self.hidden_layer = None
        self.output = np.zeros(y.shape)
        self.size_of_hidden_layer = size_of_hidden_layer
        self.learning_rate = 0.001

        # Add one bias input
        self.training_set_inputs = self.__add_bias(x)

        # The weight matrix of hidden layer
        # Add one bias weight to the layer
        # values in the range -1 to 1  and mean 0.
        self.weights1 = 2 * np.random.random((self.training_set_inputs.shape[1], size_of_hidden_layer)) - 1

        # The weight matrix of output layer, with one bias weight
        self.weights2 = 2 * np.random.random((size_of_hidden_layer, 1)) - 1

    def set_weights(self, weights1, weights2):
        self.weights1 = weights1
        self.weights2 = weights2

    def __sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return self.__sigmoid(x) * (1. - self.__sigmoid(x))

    def __relu(self, x):
        return np.maximum(x, 0)

    def __relu_derivative(self, x):
        return 1. * (x > 0)

    def __think_in_hidden_layer(self, inputs):
        self.__add_bias(inputs)
        pass

    def MSE(self):
        temp = 0.5 * ((self.y - self.output) ** 2)
        return np.average(temp)

    def __add_bias(self, inputs):
        '''
        Add one column of bias input
        :param inputs:
        :return:
        '''
        # Add [-1, -1, -1,...,-1] to the first column as bias input
        output = np.insert(inputs, 0, -1, axis=1)
        return output

    def feedfoward(self):
        # feedforward the network
        self.hidden_layer = self.__relu(np.dot(self.training_set_inputs, self.weights1))
        self.output = self.__sigmoid(np.dot(self.hidden_layer, self.weights2))

    def __print_MSE(self, iteration):
        print("Iteration #" + str(iteration) + ":")
        print("MSE:")
        print(self.MSE())

    def __write_MSE(self, filestream, iteration):
        filestream.write("Iteration #" + str(iteration) + " " + "MSE: " + str(self.MSE()) + '\n')

    def backprop(self):
        err = self.y - self.output

        # the delta in output layer
        delta_output_layer = err * self.__sigmoid_derivative(self.output)
        d_weights2 = self.learning_rate * np.dot(self.hidden_layer.T, delta_output_layer)

        # the delta in hidden layer
        delta_hidden_layer = self.__relu_derivative(self.hidden_layer) * np.dot(delta_output_layer, self.weights2.T)
        d_weights1 = self.learning_rate * np.dot(self.training_set_inputs.T, delta_hidden_layer)

        # update the weights  with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, number_of_training_iteration=None):
        # Create the current time stamp
        timestamp = time.strftime("%m%d_%H%M%S")
        # store the MSE after each iteration
        results = []
        iteration = 0
        try:
            # If does't specify the number of training iteration
            if number_of_training_iteration is None:
                with open(os.path.join('training_results/', timestamp + '.txt'), 'w') as file:
                    file.write('The number of hidden layers: ' + str(self.size_of_hidden_layer) + '\n')
                    file.write('The training rate is :' + str(self.learning_rate))
                    self.feedfoward()
                    self.__print_MSE(iteration)
                    mse = -1
                    self.__write_MSE(file, iteration)
                    # keep training until the difference of MSE less than a very small number
                    while abs(self.MSE() - mse) > 1e-11:
                        old_mse = self.MSE()
                        mse = old_mse
                        self.backprop()
                        self.feedfoward()
                        iteration += 1
                        self.__print_MSE(iteration)
                        self.__write_MSE(file, iteration)
                    self.__write_MSE(file, iteration)

            # If have the number of training iteration
            else:
                with open(os.path.join('training_results/', timestamp + '.txt'), 'w') as file:
                    file.write('The number of hidden layers: ' + str(self.size_of_hidden_layer) + '\n')
                    for iteration in range(number_of_training_iteration):
                        self.feedfoward()
                        self.__print_MSE(iteration)
                        self.backprop()
                        self.__write_MSE(file, iteration)
        # If use ctrl + C to terminate the training process, save the current results
        except KeyboardInterrupt:
            # Save the training results of weights1 and weights2 with current timestamp
            np.save(os.path.join('training_results/', timestamp) + '_weights1', self.weights1)
            np.save(os.path.join('training_results/', timestamp) + '_weights2', self.weights2)

        # Save the training results of weights1 and weights2 with current timestamp
        np.save(os.path.join('training_results/', timestamp) + '_weights1', self.weights1)
        np.save(os.path.join('training_results/', timestamp) + '_weights2', self.weights2)

        print('Timestamp: ' + timestamp)

    def estimate(self, X):
        '''
        Using the current weight1 and weight to estimate the output of test sets
        :param X: The test inputs
        :return: The estimated output
        '''
        X = self.__add_bias(X)
        h = self.__relu(np.dot(X, self.weights1))
        output = self.__sigmoid(np.dot(h, self.weights2))
        return output


if __name__ == '__main__':
    # PreProcess the inputs
    positive_training_files = glob.glob('stores/hog_descriptor/train_positive/*.npy')
    negative_training_files = glob.glob('stores/hog_descriptor/train_negative/*.npy')
    positive_training_set = []
    negative_training_set = []

    for file_path in positive_training_files:
        temp = np.load(file_path)
        positive_training_set.append(temp)

    # Import the train_positive training sets
    positive_training_set = np.asarray(positive_training_set)
    positive_label = np.ones((positive_training_set.shape[0], 1))

    for file_path in negative_training_files:
        temp = np.load(file_path)
        negative_training_set.append(temp)

    #  Import the train_negative training sets
    negative_training_set = np.asarray(negative_training_set)
    negative_label = np.zeros((negative_training_set.shape[0], 1))
    # Construct the training sets and labels
    training_set = np.concatenate((positive_training_set, negative_training_set), axis=0)
    labels = np.concatenate((positive_label, negative_label), axis=0)

    # Train the neural network
    nn = NeuralNetwork()
    nn.build_network(training_set, labels, 100)
    nn.train()
