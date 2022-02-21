import numpy as np
import csv

import iris_NN

csv_reader = csv.reader(open('iris.csv'))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



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

    my_NN = iris_NN.iris_NN(iris_data)
    my_NN.train()
    my_NN.test()
    my_NN.predict(my_NN.X_test)


