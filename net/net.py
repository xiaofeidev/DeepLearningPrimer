import sys, os

sys.path.append(os.pardir)
import numpy as np
from neural_network.activation import softmax
from loss.loss import cross_entropy_error


class simpleNet:
    def __init__(self):
        # 用高斯分布进行初始化
        # 一个 2 * 3 的权重参数矩阵
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        """
        预测就是将输入参数 x 与权重参数矩阵 W 相乘
        """
        return np.dot(x, self.W)

    def loss(self, x, t):
        """
        计算损失函数
        """
        z = self.predict(x)
        y = softmax(z)
        # 交叉熵误差
        loss = cross_entropy_error(y, t)
        return loss

