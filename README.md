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





