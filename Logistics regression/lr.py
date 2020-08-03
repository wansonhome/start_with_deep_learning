import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()

#print(train_set_y[0:,2])   默认从0维开始

index = 25
#print("y=" + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + " picture")

#print(train_set_y.shape)
#print(train_set_x_orig.shape)
m_train = train_set_y.shape[1]      #训练集数量
m_test = test_set_y.shape[1]        #测试集数量
num_px = train_set_x_orig.shape[1]  #尺寸

#降低维度为209行，-1表示系统计算
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
#print(train_set_x_flatten)
#print(train_set_x_flatten.shape)

#标准化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

#1. 定义模型结构（例如输入特征的数量）


def sigmoid(z):
    """
    :param z: 任意大小的R或ndarray
    :return: sigmoid（z）
    """
    s = 1 / (1 + np.exp(-z))
    return s
#测试sigmoid
#print(sigmoid(0))

#2. 初始化模型的参数
def init_with_zeros(dim):
    """
    创建(dim,1)的0向量
    :param dim:
    :return:
        w - 维度为(dim,1)的初始化向量
        b - 初始化标量
    """
    w = np.zeros(shape = (dim, 1))
    b = 0

    #使用断言来确保数据正确性（可选）
    assert(w.shape == (dim, 1))
    assert(isinstance(b, int) or isinstance(b, float))

    return w, b

def propagate(w, b, X, Y):
    """
    前后向传播的成本函数和梯度
    :param w:权重
    :param b:偏置
    :param x:训练集
    :param y:label
    :return:
        dw：w的损失梯度
        db：b的损失梯度
    """
    #样本数量
    m = X.shape[1]

    #正向传播
    A = sigmoid(np.dot(w.T, X) + b)

    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))

    #反向传播
    dw = (1 / m) * np.dot(X, (A-Y).T)
    db = (1 / m) * np.sum(A - Y)

    #断言（可选）
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    #cost = np.squeeze(cost)
    assert (cost.shape == ())

    #一种习惯，还挺不错
    grabs = {
        "dw": dw,
        "db": db
    }
    return grabs, cost
"""
#测试一下propagate

#初始化一些参数
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print(w,"\n", b,"\n", X,"\n", Y,"\n")
print(np.dot(w.T,X))
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
"""

def optimizer(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    梯度下降优化器
    :param w: 权重
    :param b: 偏置
    :param X: 训练集
    :param Y: label
    :param num_iterations:迭代次数
    :param learning_rate: 学习率
    :param print_cost: 打印cost
    :return:
        param：字典{w, b}
        grabs：包含{dw, db}即w，b梯度的字典
        cost：代价列表，用于绘图
    """
    costs = []
    for i in range(num_iterations):
        grabs, cost = propagate(w, b, X, Y)

        dw = grabs["dw"]
        db = grabs["db"]

        w = w - learning_rate * dw
        b = b-learning_rate * db

        #没迭代100次记录cost
        if i % 100 == 0:
            costs.append(cost)
        #打印csots
        if(print_cost) and (i % 100 == 0):
            print("iter: %i, cost：%f " % (i, cost))

    #转换字典（好习惯？）
    params = {
        "w": w,
        "b": b}
    grads = {
        "dw": dw,
        "db": db}
    return (params, grads, costs)
"""
#测试optimize
print("====================测试optimize====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
params , grads , costs = optimizer(w , b , X , Y , num_iterations=1000 , learning_rate = 0.009 , print_cost = True)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
"""
def predict(w, b, X):
    """
    使用回归参数预测标签
    :param w: 权重
    :param b: 偏置
    :param X: 数据集
    :return
        Y_prediction：包含x中找钱预测标签的nparray
    """

    #图片数量
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    #w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        #Y预测值中第一行大于0.5则label==1，否则为0
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0
    #断言（可选）
    assert (Y_prediction.shape == (1,m))

    return Y_prediction
"""
#测试predict
print("====================测试predict====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("predictions = " + str(predict(w, b, X)))
"""

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost = False):
    """
    调用传播函数和优化器进行构架模型
    :param X_train:训练集
    :param Y_train:训练集label
    :param X_test:测试集
    :param Y_test:测试集label
    :param num_iterations:迭代次数
    :param learning_rate:学习率
    :param print_cost:输出
    :return
        d：模型信息
    """

    w, b = init_with_zeros((X_train.shape[0]))

    parameters, grabs, costs = optimizer(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=True)

    #得到w，b
    w, b = parameters["w"], parameters["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    print("train set acc:", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("test set acc:", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations
    }

    return d

num_iter = 1000
alpha = 0.005

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iter, alpha,print_cost=True)

#绘图
costs = d['costs']
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iteration')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


learning_rates = [0.01, 0.005, 0.0005]
models = {}
for i in learning_rates:
    print("learning rates is:" + str(i))
    models[str(i)] = (model(train_set_x, train_set_y, test_set_x, test_set_y, num_iter, i, print_cost=False))
    print("\n"+"------------------------------------------------------")

for i in learning_rates:
    plt.plot(models[str(i)]["costs"], label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='best', shadow=True)
#frame = legend.get_frame()
#frame.set_facecolor('0.9')
plt.show()



