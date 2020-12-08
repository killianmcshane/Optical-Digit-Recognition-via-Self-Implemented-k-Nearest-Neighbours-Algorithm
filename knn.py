import numpy as np
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

'''
The following code creates an sklearn.datasets object of hand written digits and 
uses the pandas module to place it into a data-frame, 'df'. A k-nearest-neighbour
classifier is then called from the sklearn.neighbors module.
'''
digits = load_digits()
df = pd.DataFrame(digits['data'][0:500])
knn = KNeighborsClassifier()
selection = ''


'''
Below is the code for the menu selection process. Each selection option
contains the corresponding code for that selection.

All input is standardised to lowercase using .lower() to streamline
the comparison functionality.
'''
while not (selection.lower() == '6' or selection.lower() == 'quit'):
    selection = input('Please make a selection from the following: \n'
                      '    1) Data-set Details\n'
                      '    2) Train Sci-kit Learn Model\n'
                      '    3) Train Self Implemented Model\n\n'
                      '    Note: please train both models before selecting the following options: \n'
                      '    4) Accuracy Comparison\n    5) Query Models\n    6) Quit\n')

    if selection.lower() == '1' or selection.lower() == 'data-set details':
        print('Optical Recognition of Handwritten Digits Data-set\n'
              '--------------------------------------------------\n\n'
              'Classes: 10 (where each class refers to a digit, 0 to 9.)\n'
              'Total Samples: 1797 (this has been reduced to 500 to help with computation speed.)\n'
              'Attributes: 64 pixels per sample.\n'
              'Samples Per Class: ~180.\n'
              'Features: 8x8 images of pixels in the integer range 0 to 16.\n'
              'Instances: 5620.\n'
              'Train/Test Split: 70%/30%.\n'
              '--------------------------------------------------\n\n')


    elif selection.lower() == '2' or selection.lower() == 'sci-kit learn model':
        X = digits['data'][0:500]  # FEATURES
        y = digits['target'][0:500]  # LABELS

        'The function train_test_split enables the data-set to be '
        'randomly broken into separate testing and training sets. '
        X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(X, y, test_size=0.3)

        'The fit function is what enables the training of the algorithm.'
        knn.fit(X_train_sk, y_train_sk)

        'The pickle module allows for model persistence.'
        import pickle

        knn_pickle = open('knn.pickle', 'wb')
        pickle.dump(knn, knn_pickle)
        knn_pickle.close()

        test_accuracy = knn.score(X_test_sk, y_test_sk)
        training_accuracy = knn.score(X_train_sk, y_train_sk)

        'The data-set was reduced to only 500 samples as my computer was struggling with the full set.'
        test = np.array(digits['data'][0:500])
        test = test.reshape(len(test), -1)

        prediction = knn.predict(test)
        print('Sci-kit Learn Model Accuracy: ', '%.2f' % test_accuracy, '\n\n')



    elif selection.lower() == '3' or selection.lower() == 'self implemented model':
        '''
        The following is my own implementation of a knn algorithm.
        The random module is used to mirror sklearn's test_train_split function.
        I chose to use a seed (1997) to allow for replication of the results.
        The slice-notation is used to select from the list of indices easily
        I chose to use a training/testing split of 70%/30%.
        '''
        np.random.seed(1997)
        indices = np.random.permutation(len(df))
        '350 is 70% of the total 500 samples used.'
        n_training_samples = 350

        X_train = digits['data'][indices[:-n_training_samples]]
        X_test = digits['data'][indices[-n_training_samples:]]
        y_train = digits['target'][indices[:-n_training_samples]]
        y_test = digits['target'][indices[-n_training_samples:]]

        'The distances function determines the euclidean distance between two digits'
        def distances(digit1, digit2):
            digit1 = np.array(digit1)
            digit2 = np.array(digit2)

            return np.linalg.norm(digit1 - digit2)


        '''
        The neighbours function calculates the distance of a digit from all other digits in the set.
        It then appends these distances to the distance_list list. This list is then sorted using
        a lambda function (x: x[1]). Only the k nearest neighbours are then added to the neighbour_list.
        '''
        def neighbours(X_train, y, digit, k):
            distance_list = []
            for i in range(len(X_train)):
                digit_distance = distances(digit, X_train[i])
                distance_list.append((X_train[i], digit_distance, y[i]))
                distance_list.sort(key=lambda x: x[1])
                neighbour_list = distance_list[:k]

            return neighbour_list


        'The vote function votes to get a single result.'
        def vote(neighbour_list):
            digit_counter = Counter()
            for digit in neighbour_list:
                digit_counter[digit[2]] += 1
            result = digit_counter.most_common(1)[0][0]
            return result


        'Simple function to calculate correctly predicted test results divided by the total number.'
        def get_test_accuracy():
            correct_test = 0
            total_test = 0
            for i in range(500-n_training_samples):
                neighbour_list = neighbours(X_train, y_train, X_test[i], 2)
                if y_test[i] == vote(neighbour_list):
                    correct_test += 1
                total_test += 1
            self_implemented_test_accuracy = float('%.2f' % (correct_test/total_test))

            return self_implemented_test_accuracy


        'Simple function to calculate correctly predicted training results divided by the total number.'
        def get_train_accuracy():
            correct_train = 0
            total_train = 0
            for i in range(n_training_samples):
                neighbour_list = neighbours(X_train, y_train, X_test[i], 2)
                if y_test[i] == vote(neighbour_list):
                    correct_train += 1
                total_train += 1
            self_implemented_train_accuracy = correct_train/total_train

            return self_implemented_train_accuracy


        print('Self Implemented Model Accuracy: ', get_test_accuracy(), '\n\n')


    elif selection.lower() == '4' or selection.lower() == 'accuracy comparison':
        print('Sci-kit Learn Model\n'
              '-------------------\n'
              'Training Accuracy: ', '%.2f' % training_accuracy, '\n'
              'Testing Accuracy: ', '%.2f' % test_accuracy, '\n'
              'Percentage Difference: ', '%.2f' % ((training_accuracy - test_accuracy)*100), '%\n')

        print('Self Implemented Model\n'
              '----------------------\n'
              'Training Accuracy: ', '%.2f' % get_train_accuracy(), '\n'
              'Testing Accuracy: ', '%.2f' % get_test_accuracy(), '\n'
              'Percentage Difference: ', '%.2f' % ((get_train_accuracy() - get_test_accuracy())*100), '%\n\n')



    elif selection.lower() == '5' or selection.lower() == 'query model':
        index_selection = input('\nEnter an index from 350 to 499 to query the model on the test data-set:\n')
        saved_pickle = open('knn.pickle', 'rb')
        knn_saved = pickle.load(saved_pickle)
        print('The model predicted the digit as: ', knn_saved.predict(X[[int(index_selection)]]), '\n')
        print('Its corresponding real label was: ', y[[int(index_selection)]], '\n\n')

