import numpy as np
import os
import time
import glob
import sys
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    nn = NeuralNetwork()
    # Load the descriptors of test sets
    positive_test_files = glob.glob('stores/hog_descriptor/test_positive/*.npy')
    negative_test_files = glob.glob('stores/hog_descriptor/test_negative/*.npy')

    results_path = 'training_results/'

    # We need to load the weights of training results to our neural network
    #
    timestamp_of_training = '1212_204309'

    # If you specify a certain training results with a timestamp
    if len(sys.argv) != 0:
        timestamp_of_training = sys.argv[1]

    # Import the weights from the file with timestamp
    weights1 = np.load(os.path.join(results_path, timestamp_of_training + '_weights1.npy'))
    weights2 = np.load(os.path.join(results_path, timestamp_of_training + '_weights2.npy'))

    # Load the weights from the results of  training
    nn.set_weights(weights1, weights2)

    # Estimate the output of positive test sets
    estimated_test_results = []
    for file in positive_test_files:
        result = {}
        temp = np.load(file)
        x = temp.reshape((1, len(temp)))
        result['file_name'] = os.path.split(file)[1]
        output = nn.estimate(x)
        result['output'] = output[0][0]
        result['label'] = '1'
        if result['output'] < 0.5:
            result['class'] = 'Not Human'
        else:
            result['class'] = 'Human'
        estimated_test_results.append(result)

    # Estimate the output of negative test sets
    for file in negative_test_files:
        result = {}
        temp = np.load(file)
        x = temp.reshape((1, len(temp)))
        result['file_name'] = os.path.split(file)[1]
        output = nn.estimate(x)
        result['output'] = output[0][0]
        result['label'] = '0'
        if result['output'] < 0.5:
            result['class'] = 'Not Human'
        else:
            result['class'] = 'Human'
        estimated_test_results.append(result)

    timestamp = time.strftime("%m%d_%H%M%S")
    #  Print and save the test results
    with open('test_results/' + timestamp + '.txt', 'w') as file:
        print('The neural network is using the training results from ' + timestamp_of_training + '\n')
        file.write('The neural network is using the training results from ' + timestamp_of_training + '\n')
        for result in estimated_test_results:
            str2 = 'Actual Label: ' + str(result['label']) + '  Estimated Output: ' + str(result['output'])
            print(result['file_name'])
            print(str2)
            print('Estiamted Class: ' + result['class'])
            print('')
            file.write('File Name: ' + result['file_name'] + '\n')
            file.write(str2 + '\n')
            file.write('Classification: ' + result['class'] + '\n')
            file.write('\n')

    print('Timestamp: ' + timestamp)
