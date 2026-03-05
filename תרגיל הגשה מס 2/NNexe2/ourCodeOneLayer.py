# ================================================================== #
#                   Exercise 2: Neural Networks                      #
# ================================================================== #
# Submitted by: Elyasaf Cohen 311557227 and Yakir Yohanan 312252034  #
# ================================================================== #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ************************** activation functions ************************** #

# ====  sigmoid function - return value between 0 and 1 ==== #
def sigmoid(z):
    X = np.exp(z)
    return X / (1 + X)


# =========== relu function - return max (0,Z) ============= #
def relu(z):
    return np.maximum(0, z)


# ************** the derivatives of the activation functions *************** #

# ============ the derivative of sigmoid function =========== #
def sigmoid_der(z):
    A = sigmoid(z)
    return A * (1 - A)


# ============ the derivative of relu function =========== #
def relu_der(z):
    return (z > 0).astype(float)


# ============ the derivative of relu function =========== #

def tanh_der(z):
    X = np.tanh(z)
    return 1 - X ** 2


# ============ LogLoss function =========== #
def LogLoss_calculation(A, Y):
    cost = np.mean(-(Y * np.log(A) + (1 - Y) * np.log(1 - (A))))
    return cost


def activation_function(Z1, activation_faction_name):

    if activation_faction_name == "sigmoid":
        A1 = sigmoid(Z1)

    elif activation_faction_name == "tanh":
        A1 = np.tanh(Z1)

    elif activation_faction_name == "relu":
        A1 = relu(Z1)

    return A1


# ======= initialize the parameters of our artificial neural network. ========== #

# Explanation of creating weight matrices W1 and W2:
# We use np.random.randn(m, n) to create matrices of random values from a normal distribution.
# Multiplying by 0.01 limits the values to a small range, which can help in efficient learning.

# For example, if node_in_input_leyer is 4:
# W1 = np.random.randn(5, 4) * 0.01  # Creates W1 for a hidden layer with 5 units and 4 input features (5X4).
# W2 = np.random.randn(3, 5) * 0.01  # Creates W2 for an output layer with 3 units and 5 hidden units (3X5).
# ------------------------------------------------------------------------------- #
# Explanation of creating bias vectors b1 and b2:
# We use np.zeros((m, 1)) to create vectors of zeros.
# These bias vectors are added to the units in each layer.

# For example, if we have 5 units in the hidden layer and 3 units in the output layer:
# b1 = np.zeros((5, 1))  # Creates a bias vector b1 with 5 zeros for the hidden layer.
# b2 = np.zeros((3, 1))  # Creates a bias vector b2 with 3 zeros for the output layer.

def initialize_parameters(node_in_input_leyer, node_in_hidden_leyer, node_in_output_leyer):
    return {
        "W1": np.random.randn(node_in_hidden_leyer, node_in_input_leyer) * 0.01,
        "W2": np.random.randn(node_in_output_leyer, node_in_hidden_leyer) * 0.01,
        "b1": np.zeros([node_in_hidden_leyer, 1]),
        "b2": np.zeros([node_in_output_leyer, 1]),
    }


# ================================= forward propagation =================================== #
def forward_propagation(input, parameters, activation_faction_name):
    # ============= Hidden Layer ========== #
    Z1 = parameters["W1"].dot(input) + parameters["b1"]

    # ======= change here for the targil =========== #
    A1 = activation_function(Z1, activation_faction_name)

    # ============= Output Layer ========== #
    Z2 = parameters["W2"].dot(A1) + parameters["b2"]
    A2 = sigmoid(Z2)

    # Store values in cache for later use in the backward_propagation #
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


# ================================= backward propagation =================================== #
def backward_propagation(parameters, cache, X, Y):
    number_of_samples = X.shape[1]  # "X is a Cartesian representation of the dataset rows and columns
    # hence it has the number of samples in index 1,
    # for example if X.shape is (5,7), X.shape[1] will be 7.

    # ============ Assuming the Log_loss function is used like last time: ============ #

    # ============== the Output layer ============== #
    dZ2 = cache["A2"] - Y  # for the sigmoid layer
    dW2 = (1 / number_of_samples) * dZ2.dot(cache["A1"].T)
    db2 = (1 / number_of_samples) * np.sum(dZ2)

    '''
    Output Layer, using MSE cost:

    dA2 = - 1 * (Y- cache["A2"]) => The derivative of MSE is -(Y-YP) (derivative of cost), this is in slide 54
    dZ2 = dA2 * sigmoid_der(cache["Z2"]) => (node derivative * input derivative)
    dW2 = (1 / number_of_samples) * np.dot(dZ2,cache["A1"].T ) => for the input:
                                                                  A1 is the input to the second level, 
                                                                  as X is the input to the first level
    db2 = (1 / number_of_samples) * np.sum(dZ2)
    '''
    # ============== the Hidden Layer ============== #
    dA1 = np.dot(parameters["W2"].T, dZ2)
    dZ1 = dA1 * relu_der(cache["Z1"])
    dW1 = (1 / number_of_samples) * np.dot(dZ1, X.T)
    db1 = (1 / number_of_samples) * np.sum(dZ1)

    # db1 = (1 / number_of_samples) * np.sum(dZ1, axis=1, keep dims = True)
    return {
        "dW1": dW1,
        "dW2": dW2,
        "db1": db1,
        "db2": db2
    }


def update_parameters(parameters, grads, learning_rate):
    return {
        "W1": parameters["W1"] - learning_rate * grads["dW1"],
        "W2": parameters["W2"] - learning_rate * grads["dW2"],
        "b1": parameters["b1"] - learning_rate * grads["db1"],
        "b2": parameters["b2"] - learning_rate * grads["db2"],
    }


# The "nn_model" function performs the training process of a neural network.
# It includes steps of forward propagation, computing the cost function,
# backward propagation, and updating the parameters.
def nn_model(X, Y, iterations, learning_rate, number_of_nodes, activation_faction_name):
    node_in_input_leyer = X.shape[0]  # number of features in X
    node_in_hidden_leyer = number_of_nodes
    node_in_output_leyer = 1

    # === initialize the dictionary "parameters" ======= #
    parameters = initialize_parameters(node_in_input_leyer, node_in_hidden_leyer, node_in_output_leyer)

    # ======= The training loop ======= #
    for i in range(1, iterations + 1):
        # ============ forward propagation ============== #
        A2, cache = forward_propagation(X, parameters, activation_faction_name)
        # cost = MSE_calculation(A2,Y)
        cost = LogLoss_calculation(A2, Y)

        # ============ backward propagation ============== #
        gradients = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, gradients, learning_rate)
        costs.append(cost)

    return parameters, costs


# ====== the predict itself ======= #
def predict(X, parameters, activation_faction_name):
    A2, cache = forward_propagation(X, parameters, activation_faction_name)
    return np.rint(A2)  # rounds the output predictions A2 to the nearest integer using np.rint()


def prediction_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# ========================== kind of main - Experiment on an online dataset ======================== #
url = 'https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true'
# url = 'https://github.com/rosenfa/nn/blob/master/class2/spam.csv?raw=true'

data_frame = pd.read_csv(url, header=0)

# separate the features from the labels (Outcome),
# and normalize the features so that they have a mean of 0
# and a standard deviation of 1
features = data_frame.drop(['Outcome'], axis=1)
features = ((features - features.mean()) / features.std())

X = np.array(features)
Y = np.array(data_frame['Outcome'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

sk_model = LogisticRegression()
sk_model.fit(X_train, Y_train)
accuracy = sk_model.score(X_test, Y_test)

# Reshape the training and test data
# so that the rows are the features
# and the columns are the samples.
X_train, X_test = X_train.T, X_test.T

num_iterations_list = [500, 1000, 1500, 2000]  # number of iterations
alpha = 1  # learning rate

# ============================= our code for the exercise  ================================= #
act_table_train_list = []
act_table_test_list = []

plt.figure(figsize=(20, 12))  # Set figure size

for activation_faction_name in ["sigmoid", "tanh", "relu"]:
    row_train_list = []
    row_test_list = []

    for i in range(1, 7):

        columns_train_list = []
        columns_test_list = []

        for num_iterations in num_iterations_list:
            costs = []  # create an empty list that will save the training cost in each iteration.

            # ===== train the model ===== #
            parameters, costs = nn_model(X_train, Y_train, num_iterations, alpha, i, activation_faction_name)

            # === predicting training data === #
            Y_train_predict = predict(X_train, parameters, activation_faction_name)
            train_acc = prediction_accuracy(Y_train_predict, Y_train)

            # === predicting testing data === #
            Y_test_predict = predict(X_test, parameters, activation_faction_name)
            test_acc = prediction_accuracy(Y_test_predict, Y_test)

            # ==== Keep the training and testing accuracy within the parameters ==== #
            parameters["train_accuracy"] = train_acc
            parameters["test_accuracy"] = test_acc

            # == Save training and test accuracy for plotting == #
            columns_train_list.append(train_acc)
            columns_test_list.append(test_acc)

        row_train_list.append(columns_train_list)
        row_test_list.append(columns_test_list)

        # ========== Plotting all costs in one plot ============ #
        plt.plot(num_iterations_list, columns_train_list, label=f'{activation_faction_name} train - {i} nodes',
                 linewidth=2)
        plt.plot(num_iterations_list, columns_test_list, label=f'{activation_faction_name} test - {i} nodes',
                 linewidth=2)

    train_table = pd.DataFrame(row_train_list)
    test_table = pd.DataFrame(row_test_list)

    train_table.columns = ["500 iter", "1000 iter", "1500 iter", "2000 iter"]
    test_table.columns = ["500 iter", "1000 iter", "1500 iter", "2000 iter"]

    print("train " + activation_faction_name + " table (alpha = " + str(alpha) + "):")
    print(train_table, end="\n\n")

    print("test " + activation_faction_name + " table (alpha = " + str(alpha) + "):")
    print(test_table, end="\n\n\n")

# =========== Set font size ============= #
plt.rcParams.update({'font.size': 16})

# ========== Display all training and test accuracies on one plot ========== #
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy over Iterations')
plt.legend(bbox_to_anchor=(0, 0.5), loc='center left', fontsize='xx-small', ncol=2)
plt.grid(True)  # Add grid
plt.show()

print("Logistic Regression accuracy:", accuracy)
