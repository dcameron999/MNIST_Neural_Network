import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore

SCALE_FACTOR = 255 # MNIST data has vales from 0 to 255.  This scales them down to a number from 0 to 1.  Don't change
LAYER0_SIZE = 784 # This is the size of each image.  Don't change.
LAYER1_SIZE = 30
LAYER2_SIZE = 20
LAYER3_SIZE = 10
TRAINING_ITERATIONS = 500
TRAINING_RATE = 0.5
TRAIN_DATA_FILE = 'MNIST_CSV/mnist_train.csv'
TEST_DATA_FILE = 'MNIST_CSV/mnist_test.csv'
NUMBER_OF_TESTS = 5000
SHOW_IMAGES = False # Set to True to draw the image of the incorrect predictions.  Not recommended if NUMBER_OF_TESTS > 50

# This is a simple neural network for MNIST hand written numbers
# Layer 0 (input)       Layer 1 (hidden)    Layer 2 (hidden)        Layer 3 (output)
# 784 nodes             30 nodes            20 nodes                10 nodes
#
#   O
#   O                   0
#   O                   O                   O                       0
#   O                   O                   O                       0
#   O                   O                   O                       0
#   .                   O                   O                       0
#   .                   .                   .                       0
#   .                   .                   .                       0
#   O                   O                   O                       0
#   O                   O                   O                       0
#   O                   O                   O                       0
#   O                   O                   O                       0
#   O                   0
#   O
#
# Forward Propagation
# A0 = input layer (file data)
# Z1 = W1 * A0 + b1 = unactivated first layer (where W1 = weights for layer 1, b1 = biases for layer 1)
# A1 = ReLU(Z1) = first layer - activation function is ReLU
# Z2 = W2 * A1 + b2 = unactivated second layer (where W2 = weights for layer 2, b2 = biases for layer 2)
# A2 = softmax(Z2) = second layer - activation function is Softmax
#
# Back Propagation
# dZ2 = A2 - Y = error of the second layer (where Y is the actual labels one hot encoded)
# dW2 = 1 / m * dZ2 * A1 (where m is the number of training examples)
# db2 = 1 / m * sum dZ2 (where m is the number of training examples)
# dZ1 = W2 * dZ2 * g = error of the first layer (where g is the derivative of the activation function)
# dW1 = 1 / m * dZ1 * A0 (where m is the number of training examples)
# db2 = 1 / m * sum dZ1 (where m is the number of training examples)
#
# Update Parameters
# W1 = W1 - alpha * dW1 (where alpha is the learning rate)
# b1 = b1 - alpha * db1 (where alpha is the learning rate)
# W2 = W2 - alpha * dW2 (where alpha is the learning rate)
# b2 = b2 - alpha * db2 (where alpha is the learning rate)
#
#

def draw_mnist_image(data_array,label,prediction_is_correct, prediction):
    # Reshape the array into 28 x 28 array (2-dimensional array)
    data_array = data_array.T.reshape((28, 28))
    if prediction_is_correct:
        plt.title('Label is {label} - Prediction of '.format(label=label) + str(prediction) + ' is Correct')
    else:
        plt.title('Label is {label} - Prediction of '.format(label=label) + str(prediction) + ' is Incorrect')
    plt.imshow(data_array, cmap='gray')
    plt.show()


def initialize_parameters():
    W1 = np.random.rand(LAYER1_SIZE,LAYER0_SIZE) - 0.5
    b1 = np.random.rand(LAYER1_SIZE,1) - 0.5
    W2 = np.random.rand(LAYER2_SIZE,LAYER1_SIZE) - 0.5
    b2 = np.random.rand(LAYER2_SIZE,1) - 0.5
    W3 = np.random.rand(LAYER3_SIZE,LAYER2_SIZE) - 0.5
    b3 = np.random.rand(LAYER3_SIZE,1) - 0.5
    return W1,b1,W2,b2,W3,b3

def ReLU(Z):
    return np.maximum(0, Z)


def derivative_of_ReLU(Z):
    return Z > 0


def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def forward_propagation(W1, b1, W2, b2, W3, b3, X):
    # Forward Propagation
    # A0 = input layer (file data)
    # Z1 = W1 * A0 + b1 = unactivated first layer (where W1 = weights for layer 1, b1 = biases for layer 1)
    # A1 = ReLU(Z1) = first layer - activation function is ReLU
    # Z2 = W2 * A1 + b2 = unactivated second layer (where W2 = weights for layer 2, b2 = biases for layer 2)
    # A2 = softmax(Z2) = second layer - activation function is Softmax
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def backwards_propagation(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, one_hot_Y, m):
    # Back Propagation
    # dZ2 = A2 - Y = error of the second layer (where Y is the actual labels one hot encoded)
    # dW2 = 1 / m * dZ2 * A1 (where m is the number of training examples)
    # db2 = 1 / m * sum dZ2 (where m is the number of training examples)
    # dZ1 = W2 * dZ2 * g = error of the first layer (where g is the derivative of the activation function)
    # dW1 = 1 / m * dZ1 * A0 (where m is the number of training examples)
    # db2 = 1 / m * sum dZ1 (where m is the number of training examples)

    dZ3 = 2 * (A3 - one_hot_Y)
    dW3 = dZ3.dot(A2.T) / m
    db3 = np.sum(dZ3,1) / m

    dZ2 = W3.T.dot(dZ3) * derivative_of_ReLU(Z2)
    dW2 = dZ2.dot(A1.T) / m
    db2 = np.sum(dZ2,1) / m

    dZ1 = W2.T.dot(dZ2) * derivative_of_ReLU(Z1)
    dW1 = dZ1.dot(X.T) / m
    db1 = np.sum(dZ1,1) / m
    return dW1, db1, dW2, db2, dW3, db3


def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (LAYER1_SIZE,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (LAYER2_SIZE,1))
    W3 -= alpha * dW3
    b3 -= alpha * np.reshape(db3, (LAYER3_SIZE,1))

    return W1, b1, W2, b2, W3, b3

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size * 100

def get_predictions(A3):
    return np.argmax(A3,0)


def gradient_descent(X, Y, iterations, alpha):
    one_hot_Y = one_hot(Y)
    m = Y.size
    W1, b1, W2, b2, W3, b3 = initialize_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backwards_propagation(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, one_hot_Y, m)
        W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if (i + 1) % (iterations / 10) == 0:
            print('Iteration: ' + Fore.LIGHTCYAN_EX + str(i + 1) + Fore.RESET + ' - ' + Fore.LIGHTWHITE_EX + '%.2f' % ((i+1)/iterations*100) + '%' + Fore.RESET + ' complete')
            print('Accuracy: ' + Fore.LIGHTGREEN_EX + '%.2f' % get_accuracy(get_predictions(A3),Y) + '%' + Fore.RESET)
    return W1, b1, W2, b2, W3, b3, get_accuracy(get_predictions(A3),Y)


def make_predictions(X, W1 ,b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def show_prediction(index,X, Y, W1, b1, W2, b2, W3, b3):
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2, W3, b3)
    label = Y[index]
    if label == prediction[0]:
        print("Predicted Number: " + Fore.LIGHTGREEN_EX + str(prediction[0]) + Fore.RESET + '  Actual Number: ' + Fore.LIGHTCYAN_EX + str(label) + Fore.RESET)
        return True
    else:
        print("Predicted Number: " + Fore.LIGHTRED_EX + str(prediction[0]) + Fore.RESET + '  Actual Number: ' + Fore.LIGHTCYAN_EX + str(label) + Fore.RESET)
        if SHOW_IMAGES:
            draw_mnist_image(X.T[index], label, False,prediction[0])
        return False




#### MAIN CODE ####

users_choice = input('Press Y to train the model or N to jump to testing the model (assumes training has already been performed)')
if users_choice == 'y':
    print('Training the model now...')
    # Set up the data we need for training and testing
    dataframe_train = pd.read_csv(filepath_or_buffer=TRAIN_DATA_FILE, header=None)
    data_train = np.array(dataframe_train)
    np.random.shuffle(data_train)
    data_train = data_train.T
    labels_train = data_train[0]
    data_train = data_train[1:len(data_train)] / SCALE_FACTOR

    # Train the model
    W1, b1, W2, b2, W3, b3, trained_accuracy = gradient_descent(X=data_train, Y=labels_train, iterations=TRAINING_ITERATIONS, alpha=TRAINING_RATE)

    # Store the trained model parameters to file
    with open("trained_model_parameters2.pkl","wb") as trained_parameters_file:
        pickle.dump((W1, b1, W2, b2, W3, b3, trained_accuracy), trained_parameters_file)


with open("trained_model_parameters2.pkl","rb") as trained_parameters_file:
    W1, b1, W2, b2, W3, b3, trained_accuracy = pickle.load(trained_parameters_file)

# Use the trained model to predict test data values
dataframe_test = pd.read_csv(filepath_or_buffer=TEST_DATA_FILE,header=None)
data_test = np.array(dataframe_test)
np.random.shuffle(data_test)
data_test = data_test.T
labels_test = data_test[0]
data_test = data_test[1:len(data_test)] / SCALE_FACTOR

print('This model has a trained accuracy of ' + Fore.LIGHTGREEN_EX + '%.2f' % trained_accuracy + '%' + Fore.RESET)
correct_prediction_count = 0
number_of_tests = NUMBER_OF_TESTS
for i in range(0, number_of_tests):
    prediction_is_correct = show_prediction(i,data_test,labels_test,W1,b1,W2,b2, W3, b3)
    correct_prediction_count += prediction_is_correct

test_accuracy = correct_prediction_count / number_of_tests * 100
print('The testing accuracy for ' + Fore.LIGHTCYAN_EX + str(number_of_tests) + Fore.RESET + ' tests is ' + Fore.LIGHTGREEN_EX + '%.2f' % test_accuracy + '%' + Fore.RESET)
print('This model has a trained accuracy of ' + Fore.LIGHTGREEN_EX + '%.2f' % trained_accuracy + '%' + Fore.RESET)