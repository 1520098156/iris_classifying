import numpy as np
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss_function(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


class iris_NN:
    def __init__(self, iris_data):
        self.lr = 0.1  # learning rate
        self.iris_data = iris_data

        X = np.array(self.iris_data[0:, 0:4])
        y = np.array([self.iris_data[0:, 4]])
        y = y.T
        X = X / 8 * 0.99 + 0.01  # 归一化
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y)
        self.X_train = X_train.T
        self.X_test = X_test.T

        self.a2 = np.zeros((3, 38))

        self.W1 = np.random.normal(0.0, 1, (8, 4))
        self.W2 = np.random.normal(0.0, 1, (3, 8))
        self.B1 = np.zeros((8, 1))
        self.B2 = np.zeros((3, 1))

    def train(self):
        oneHot = np.identity(3)
        for i in range(oneHot.shape[0]):
            for j in range(oneHot.shape[1]):
                if oneHot[i, j] == 1:
                    oneHot[i, j] = 0.99
                else:
                    oneHot[i, j] = 0.01

        y_true = oneHot[self.y_train.T.astype(int)][0]
        y_true = y_true.T

        W1 = np.random.normal(0.0, 1, (8, 4))
        W2 = np.random.normal(0.0, 1, (3, 8))
        B1 = np.zeros((8, 1))
        B2 = np.zeros((3, 1))



        for i in range(5000):
            out1 = np.dot(W1, self.X_train) + B1
            act1 = sigmoid(out1)
            out2 = np.dot(W2, act1) + B2
            act2 = sigmoid(out2)

            dZ2 = act2 - y_true
            dW2 = 1 / 112 * np.dot(dZ2, act1.T)
            dB2 = 1 / 112 * np.sum(dW2, axis=1, keepdims=True)

            dZ1 = np.dot(W2.T, dZ2) * (act1 * (1 - act1))
            dW1 = 1 / 112 * np.dot(dZ1, self.X_train.T)
            dB1 = 1 / 112 * np.sum(dZ1, axis=1, keepdims=True)

            W2 -= self.lr * dW2
            B2 -= self.lr * dB2
            W1 -= self.lr * dW1
            B1 -= self.lr * dB1

        self.W2 = W2
        self.B2 = B2
        self.W1 = W1
        self.B1 = B1

        o1 = np.dot(W1, self.X_test) + B1
        a1 = sigmoid(o1)
        o2 = np.dot(W2, a1) + B2
        self.a2 = sigmoid(o2)

    def test(self):
        result = []

        for i in range(self.a2.T.shape[0]):
            result.append(np.argmax(self.a2.T[i]))

        true_no = 0
        for i in range(len(result)):
            if result[i] == self.y_test[i][0]:
                true_no += 1
        print('Correct rate =', true_no / len(result) * 100, '%')

    def predict(self, X_predict):
        result = []

        out1 = np.dot(self.W1, X_predict) + self.B1
        act1 = sigmoid(out1)
        out2 = np.dot(self.W2, act1) + self.B2
        act2 = sigmoid(out2)

        for i in range(act2.T.shape[0]):
            result.append(np.argmax(act2.T[i]))

        print('Prediction is:')
        print(result)
        print('\'0\' means setosa\n\'1\' means versicolor\n\'2\' means virginica')
