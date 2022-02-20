import numpy as np
from sklearn.model_selection import train_test_split
import csv

csv_reader = csv.reader(open('iris.csv'))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss_function(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


if __name__ == '__main__':
    iris_data = []

    for row in csv_reader:
        if row[5] == 'setosa':
            row[5] = 0
        elif row[5] == 'versicolor':
            row[5] = 1
        else:
            row[5] = 2
        iris_data.append(row[1:])
    iris_data.pop(0)
    iris_data = np.array(iris_data, dtype='float')

    iris_data = np.random.permutation(iris_data)

    X = np.array(iris_data[0:, 0:4])

    y = np.array([iris_data[0:, 4]])
    y = y.T

    X = X / 8 * 0.99 + 0.01  # 归一化

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = X_train.T
    X_test = X_test.T

    oneHot = np.identity(3)
    for i in range(oneHot.shape[0]):
        for j in range(oneHot.shape[1]):
            if oneHot[i, j] == 1:
                oneHot[i, j] = 0.99
            else:
                oneHot[i, j] = 0.01

    y_true = oneHot[y_train.T.astype(int)][0]
    y_true = y_true.T

    W1 = np.random.normal(0.0, 1, (8, 4))
    W2 = np.random.normal(0.0, 1, (3, 8))
    B1 = np.zeros((8, 1))
    B2 = np.zeros((3, 1))

    lr = 0.1

    for i in range(5000):
        out1 = np.dot(W1, X_train) + B1
        act1 = sigmoid(out1)
        out2 = np.dot(W2, act1) + B2
        act2 = sigmoid(out2)

        dZ2 = act2 - y_true
        dW2 = 1 / 112 * np.dot(dZ2, act1.T)
        dB2 = 1 / 112 * np.sum(dW2, axis=1, keepdims=True)

        dZ1 = np.dot(W2.T, dZ2) * (act1 * (1 - act1))
        dW1 = 1 / 112 * np.dot(dZ1, X_train.T)
        dB1 = 1 / 112 * np.sum(dZ1, axis=1, keepdims=True)

        W2 -= lr * dW2
        B2 -= lr * dB2
        W1 -= lr * dW1
        B1 -= lr * dB1

        # loss = loss_function(act2, y_true)
        # print(np.sum(loss))

    o1 = np.dot(W1, X_test) + B1
    a1 = sigmoid(o1)
    o2 = np.dot(W2, a1) + B2
    a2 = sigmoid(o2)

    result = []

    for i in range(a2.T.shape[0]):
        result.append(np.argmax(a2.T[i]))

    true_no = 0
    for i in range(len(result)):
        if result[i] == y_test[i][0]:
            true_no += 1
    print('Correct rate =',true_no / len(result) * 100, '%')
