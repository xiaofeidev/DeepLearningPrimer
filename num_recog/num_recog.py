# 手写数字识别
# 假设学习已经全部结束(已获得全部的权重和偏置参数值)
# 我们使用学习到的参数，先实现神经网络的【推理处理】
# 学习到的权重参数保存在 pickle 文件 sample_weight.pkl 中
# 这个文件中以字典变量的形式保存了权重和偏置参数

'''
本章的所有实现均使用 MNIST 手写数字图像集
MNIST数据集是由0到9的数字图像构成的
训练图像有6万张，测试图像有1万张，这些图像可以用于学习和推理
MNIST的图像数据是28像素 × 28像素（共 784 个像素）的灰度图像（1通道），各个像素的取值在0到255之间
每个图像数据都相应地标有“7”“2”“1”等标签
---
神经网络的输入层有784个神经元，输出层有10个神经元。输入层的784这个数字来
源于图像大小的28 × 28 = 784，输出层的10这个数字来源于10类别分类（数
字0到9，共10个类别）。此外，这个神经网络有2个隐藏层，第1个隐藏层有
50个神经元，第2个隐藏层有100个神经元。这个50和100可以设置为任何值
'''

import os
import pickle
import numpy as np
from data.mnist import load_mnist, load_mnist_test
# from ..neural_network.activation import *
# from neural_network.activation import *
import sys

# sys.path.append("..")
# sys.path.append(os.pardir)

# 获取当前文件所在【目录】的绝对路径，注意是当前文件的所在目录，而不是当前文件
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录的绝对路径（假设上一级就是根目录），上一级确实刚好就是本项目的根目录
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到包查找路径中，很重要，不然会找不到这个项目中的其他自定义模块，也就是会报导入错误
sys.path.append(project_root)

from neural_network.activation import *


# 获取训练数据，训练标签；测试数据，测试标签
def get_data():
    (x_test, t_test) = \
        load_mnist_test(normalize=True, flatten=True, one_hot_label=True)
    # normalize 设置为 True，即实现输入数据【正规化】
    # 这里将各个像素值除以255，使得数据的值在0.0～1.0的范围内
    # 神经网络的输入数据进行某种既定的转换称为【预处理】
    # 这里对输入图像数据的预处理即正规化
    return x_test, t_test


# 初始化已经训练好的神经网络(神经网络的的所有参数已训练好并缓存了起来)
# 缓存逻辑通过 python 的 pickle 机制实现
def init_network():
    # sample_weight.pkl 文件以字典变量的形式保存了权重和偏置参数
    print(os.getcwd())  # 获取当前工作目录路径
    relaPath = os.path.join('data', "md_2023-11-13_10-47-13.pkl")
    absPath = os.path.join(current_dir, relaPath)
    with open(absPath, 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    # 各中间层的权重参数
    W1, W2 = network['W1'], network['W2']
    # 各中间层的偏置参数
    b1, b2 = network['b1'], network['b2']
    # 输入层到第一个隐层
    a1 = np.dot(x, W1) + b1
    z1 = relu(a1)  # 隐层激活函数
    # 第一个隐层到第二个隐层
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)  # 输出层激活函数

    return y


# 这个是使用书中已训练好的 sample_weight.pkl 模型文件做推理的代码
# def predict(network, x):
#     # 各中间层的权重参数
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     # 各中间层的偏置参数
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
#     # 输入层到第一个隐层
#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)  # 隐层激活函数
#     # 第一个隐层到第二个隐层
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)  # 隐层激活函数
#     # 第二个隐层到输出层
#     a3 = np.dot(z2, W3) + b3
#     y = softmax(a3)  # 输出层激活函数
#
#     return y


""" 
# 神经网络的推理，并评价其识别精度，单处理版本，即一次处理一张图片
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 获取最大值的 index
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x))) 
# 当前的识别精度为 0.9352，以后我们会花精力在神经网络的结构和学习方法上，将识别精度提高到 0.99 以上
"""

# 神经网络的推理，并评价其识别精度，批处理版本，即一次处理很多张图片，这里是一次 100 张
x, t = get_data()
if t.ndim != 1:  # 不是 1 就是 2
    #  如果监督数据是 one-hot 形式，则做此处理
    #  否则不需要处理
    t = np.argmax(t, axis=1)

network = init_network()

batch_size = 100  # 批大小为 100，即一次处理 100 张图片
accuracy_cnt = 0

for i in range(0, len(x), batch_size):  # 步长为 100，即 i 的取值会是：0，100，200，...
    x_batch = x[i:i + batch_size]  # [0,100), [100,200), ...
    # 预测结果
    y_batch = predict(network, x_batch)
    # 矩阵的第 0 维是列方向，第 1 维是行方向
    # 这里的 axis 参数取 1，即取每行的最大值
    # 这样结果会是个长度为 100 的一维数组
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
