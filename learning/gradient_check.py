"""
梯度确认操作，即对比反向传播和数值微分两种方式计算出来的梯度，凭其差值来判断反向传播计算的梯度是否正确
"""
from learning import TwoLayerNet
from num_recog.data.mnist import load_mnist
import numpy as np

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 初始化一个两层神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.num_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))