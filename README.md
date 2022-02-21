# iris_classifying
## 任务目标
完成一个算法，该算法用来对[鸢尾花数据集](http://archive.ics.uci.edu/ml/datasets/Iris)进行分类。分类的正确率应超过90%。
## 任务流程
- 预处理数据
- 训练神经网络模型
- 测试神经网络模型
- 进行预测
## 预处理数据
鸢尾花数据集是由150条数据，每一条代表一束鸢尾花个体，即150朵鸢尾花构成。这150多花由3个特征变量(Species)组成：setosa, versicolor和virginica。每一朵花有Sepal.Length, Sepal.Width, Petal.Length和Petal.Width四个特征变量。  

从csv文件中提取数据集。分别用0, 1和2表示setosa, versicolor和virginica作为特征变量。
```
import csv

csv_reader = csv.reader(open('iris.csv'))
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
```
打乱训练集，以防止规整数据使神经网络过拟合。   
```
iris_data = np.random.permutation(iris_data)
```
## 训练神经网络模型
本任务运用单隐藏层的全连接神经网络。  
以下是图解（自己画的）：
![C2第二轮神经网络图解](https://user-images.githubusercontent.com/52622948/154934227-2c29a318-ed22-4889-b4f7-643b6b749274.png)
### 初始化变量
将训练集归一化，提升收敛速度。分割数据，75%的数据(112条数据)被用来作为训练集，25%的数据(38条数据)用来作为测试集。   
输入为鸢尾花的四个特征变量，构成4\*112的矩阵，矩阵的每一列是一条数据代表一束鸢尾花个体。输出是鸢尾花的类别变量，并用独热方法构成3\*112的矩阵。
```
from sklearn.model_selection import train_test_split

def __init__(self, iris_data):
    self.__lr = 0.1  # learning rate
    self.__iris_data = iris_data

    X = np.array(self.__iris_data[0:, 0:4])
    y = np.array([self.__iris_data[0:, 4]])
    y = y.T
    X = X / 8 * 0.99 + 0.01  # 归一化
    X_train, X_test, self.y_train, self.y_test = train_test_split(X, y)
    self.X_train = X_train.T
    self.X_test = X_test.T
    
    oneHot = np.identity(3)
    for i in range(oneHot.shape[0]):
        for j in range(oneHot.shape[1]):
            if oneHot[i, j] == 1:
                oneHot[i, j] = 0.99
            else:
                oneHot[i, j] = 0.01

    y_true = oneHot[self.y_train.T.astype(int)][0]
    self.y_true = y_true.T

    self.a2 = np.zeros((3, 38))

    self.W1 = np.random.normal(0.0, 1, (8, 4))
    self.W2 = np.random.normal(0.0, 1, (3, 8))
    self.B1 = np.zeros((8, 1))
    self.B2 = np.zeros((3, 1))
```
### 开始训练模型
创建sigmoid函数用来作为激活函数。Sigmoid函数常被用作神经网络的激活函数，将变量映射到0,1之间。
```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

sigmoid函数的图像为：   
![image](https://user-images.githubusercontent.com/52622948/154918978-d60c4487-2522-45ab-aaa1-7f7f5d5f0614.png)   
随机初始化权重值，如果初始化为0，则多个神经元的作用相当于一个。初始化偏置值为0。   
```
W1 = np.random.normal(0.0, 1, (8, 4))
W2 = np.random.normal(0.0, 1, (3, 8))
B1 = np.zeros((8, 1))
B2 = np.zeros((3, 1))
```
#### 开始第一个epoch的过程。
正向传播。第一层输入层的输入是训练集的数据，经过全连接计算得到输入层的输出结果out1，输出结果经过sigmoid激活函数后传给隐藏层作为隐藏层的输入数据act1。   
隐藏层经过全连接计算加偏置值得到隐藏层的输出结果out2，输出结果经过激活函数后作为输出层的输入act2,输出层不做运算直接输出act2。输出层输出的就是预测概率。
```
out1 = np.dot(W1, self.X_train) + B1
act1 = sigmoid(out1)
out2 = np.dot(W2, act1) + B2
act2 = sigmoid(out2)
```
运用[链式法则](https://tutorial.math.lamar.edu/classes/calcI/ChainRule.aspx)计算梯度。  
```
dZ2 = act2 - self.y_true
dW2 = 1 / 112 * np.dot(dZ2, act1.T)
dB2 = 1 / 112 * np.sum(dW2, axis=1, keepdims=True)

dZ1 = np.dot(W2.T, dZ2) * (act1 * (1 - act1))
dW1 = 1 / 112 * np.dot(dZ1, self.X_train.T)
dB1 = 1 / 112 * np.sum(dZ1, axis=1, keepdims=True)
```
运用梯度下降的方法更新W2,W1,B2和B1梯度。使其结果靠近损失函数的最优解。学习率设为0.1而不是0.01是为了加快下降速度。   
```
W2 -= self.lr * dW2
B2 -= self.lr * dB2
W1 -= self.lr * dW1
B1 -= self.lr * dB1
```
#### 到此为第一个epoch结束。
应当进行多次的epoch使损失函数靠近最优解。   
```
# 损失函数
def loss_function(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))
```   
可以在每个epoch中调用损失函数查看收敛过程。
```
# 每两百次输出一侧损失函数
if epoch % 200 == 0:
    print(np.sum(loss_function(act2, self.y_true)))
```
![image](https://user-images.githubusercontent.com/52622948/154925274-04376242-314d-43b3-b382-edd8eb05d3cc.png)

可以看出收敛的速度还是很快的，epoch的次数越多后续的收敛速度越慢，越接近最优解，因此需要多次尝试epoch的次数，找到一个合适的次数。   
这里是进行了5000次的epoch。
```
def train(self):

    W1 = np.random.normal(0.0, 1, (8, 4))
    W2 = np.random.normal(0.0, 1, (3, 8))
    B1 = np.zeros((8, 1))
    B2 = np.zeros((3, 1))

    for epoch in range(5000):
        out1 = np.dot(W1, self.X_train) + B1
        act1 = sigmoid(out1)
        out2 = np.dot(W2, act1) + B2
        act2 = sigmoid(out2)

        dZ2 = act2 - self.y_true
        dW2 = 1 / 112 * np.dot(dZ2, act1.T)
        dB2 = 1 / 112 * np.sum(dW2, axis=1, keepdims=True)

        dZ1 = np.dot(W2.T, dZ2) * (act1 * (1 - act1))
        dW1 = 1 / 112 * np.dot(dZ1, self.X_train.T)
        dB1 = 1 / 112 * np.sum(dZ1, axis=1, keepdims=True)

        W2 -= self.lr * dW2
        B2 -= self.lr * dB2
        W1 -= self.lr * dW1
        B1 -= self.lr * dB1

        # 每两百次输出一侧损失函数
        # if epoch % 200 == 0:
        #     print(np.sum(loss_function(act2, self.y_true)))
```

最终将权重值和偏执值重新赋给类变量,完成训练
```
self.W2 = W2
self.B2 = B2
self.W1 = W1
self.B1 = B1
```

## 测试神经网络模型
拿测试集的数据测试神经网络的正确率。
```
def test(self):
    result = []

    o1 = np.dot(self.W1, self.X_test) + self.B1
    a1 = sigmoid(o1)
    o2 = np.dot(self.W2, a1) + self.B2
    a2 = sigmoid(o2)

    for i in range(a2.T.shape[0]):
        result.append(np.argmax(a2.T[i]))

    true_no = 0
    for i in range(len(result)):
        if result[i] == self.y_test[i][0]:
            true_no += 1
    print('Correct rate =', true_no / len(result) * 100, '%')
```
5000次epoch训练出来的网络正确率可以达到90以上。
```
Correct rate = 94.73684210526315 %
```

## 进行预测
由于150条的数据实在是少，我们拿测试集当作验证集来进行预测，测试集的数据集并没有参与训练，因此我们认为如果使用真正的验证集与训练集得出的结果应是相似的。
```
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
```
预测结果：
```
Prediction is:
[[1 0 0 2 1 2 2 1 0 1 2 2 0 0 2 2 1 0 1 1 1 1 2 2 0 1 0 0 1 0 0 2 2 1 0 0 2 1]]
'0' means setosa
'1' means versicolor
'2' means virginica
```
真实结果：
```
[[1 0 0 2 1 2 2 1 0 1 2 2 0 0 2 2 1 0 2 1 1 1 2 2 0 1 0 0 1 0 0 2 2 1 0 0 2 1]]
```
可以看到这次预测只预测错了一束鸢尾花的品种，正确率高于90\%。

## 主函数
主函数主要是预处理数据和调用神经网络进行训练，测试和预测的操作。
```
import numpy as np
import csv
import iris_NN


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
    y_test_true = my_NN.y_test
    y_test_true = np.array(y_test_true, dtype='int')
    print(y_test_true.T)
```

## 结语
本次的任务使用了一个简单的只有一层隐藏层的神经网络对鸢尾花数据集进行了分类。可以看到分类的效果不错，可以轻易达到90\%以上的正确率。
