import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def sigmoid(z):
    X = np.exp(z)
    return X / (1 + X)


def initialize_with_zeros(dim):
    return np.zeros((dim, 1)), 0


def propagate(w, b, X, Y):
    assert X.shape[0] == w.shape[0]  # This is n from the slides
    m = X.shape[1]
    assert X.shape[1] == Y.shape[1]  # This is the value of m from the slides

    Z = np.dot(w.T, X) + b  # Activation function for z
    assert Z.shape == (1, m)
    A = sigmoid(Z)
    dZ = A - Y  # Compare to a - y_train from last targil
    dw = 1 / m * np.dot(X, dZ.T)
    db = 1 / m * np.sum(dZ)
    # cost calculation for debugging
    cost = np.mean(-(Y * np.log(A) + (1 - Y) * np.log(1 - (A))))

    return cost, dw, db


def predict(w, b, X):
    Z = np.dot(w.T, X) + b
    A = np.rint(sigmoid(Z))
    return A


w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
# print(X.shape)
# print(w.shape)
print(predict(w, b, X))


def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):
        cost, dw, db = propagate(w, b, X, Y)
        costs.append(cost)
        w = w - learning_rate * dw
        b = b - learning_rate * db

    grads = {'dw': dw, 'db': db}

    thetas = {'w': w, 'b': b}
    return thetas, grads, costs


theta, grads, costs = optimize(np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]),
                               np.array([[1, 0, 1]]),
                               num_iterations=100, learning_rate=0.009)
print(theta, grads)


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    dim = X_train.shape[0]
    # print(':',Y_train.shape, Y_train.shape[1])
    w, b = initialize_with_zeros(dim)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    Y_hat1 = predict(params['w'], params['b'], X_train)
    Y_hat2 = predict(params['w'], params['b'], X_test)
    # print(("pred",Y_hat1),'real', Y_train)
    acc_train = 1 - np.mean(np.square(Y_hat1 - Y_train))
    acc_test = 1 - np.mean(np.square(Y_hat2 - Y_test))

    d = {'w': params['w'],
         'b': params['b'],
         'costs': costs,
         'Y_prediction_train': acc_train,
         'Y_prediction_test': acc_test,
         'num_iterations': num_iterations,
         'learning_rate': learning_rate}

    return d


X_train, Y_train, X_test, Y_test = np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]]), np.array(
    [[1., -1.1, -3.2], [1.2, 2., 0.1]]), np.array([[1, 1, 0]])

print(X_train.shape, Y_train.shape)

md = model(X_train, Y_train, X_test, Y_test,
           num_iterations=100, learning_rate=0.009)
print(md)

url = 'https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true'
# url = 'https://github.com/rosenfa/nn/blob/master/class2/spam.csv?raw=true'
df = pd.read_csv(url, header=0)

features = df.drop(['Outcome'], axis=1)
# features = df.drop(['class'], axis = 1)
features = ((features - features.mean()) / features.std())

X = np.array(features)
Y = np.array(df['Outcome'])
# Y = np.array(df['class'])
Y = Y.reshape(len(Y), 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
df

my_model = model(X_train.T, Y_train.T, X_test.T, Y_test.T, num_iterations=2000, learning_rate=0.01)

sk_model = LogisticRegression()
sk_model.fit(X_train, Y_train.ravel())  # if you don't do "ravel" it tells you to add it because of the shape
accuracy = sk_model.score(X_test, Y_test)
print("accuracy = ", accuracy * 100, "%")
# print(Y_train)

my_model = model(X_train.T, Y_train.T, X_test.T, Y_test.T, num_iterations=5000, learning_rate=0.005)

plt.plot(my_model['costs'])

print('\nTrain accuracy:', my_model['Y_prediction_train'], '\nTest accuracy:', my_model['Y_prediction_test'])
