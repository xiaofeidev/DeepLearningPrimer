"""
减小权重参数的值可以抑制过拟合的发生
因此一开始会将权重的初始值设置成很小的值
但是千万注意不能将权重初始值全设为 0，因为这会导致无法正确进行学习
"""

"""
当激活函数使用ReLU时，权重初始值使用He初始值
当激活函数为 sigmoid 或 tanh 等S型曲线函数时，初始值使用Xavier初始值
这是目前的最佳实践
"""

import numpy as np
import matplotlib.pyplot as plt

from neural_network.activation import *

input_data = np.random.randn(1000, 100)  # 1000个数据
node_num = 100  # 各隐藏层的节点（神经元）数
hidden_layer_size = 5  # 隐藏层有5层
activations = {}  # 激活值的结果保存在这里

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # 改变初始值进行实验！注意下面四个参数初始化方法均使用高斯分布，只是标准差不同
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01

    # Xavier 初始值，适用于 S 型函数(Sigmoid, 双曲正切函数)
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    # He 初始值，适用于 ReLU 函数
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x, w)

    # 将激活函数的种类也改变，来进行实验！
    # z = sigmoid(a)
    z = relu(a)
    # z = tanh(a)

    activations[i] = z

# 绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()