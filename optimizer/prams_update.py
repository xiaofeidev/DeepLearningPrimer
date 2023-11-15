"""
神经网络中的参数的更新的相关优化工作
"""
import numpy as np


class SGD:
    """
    SGD 即 stochastic gradient descent 随机梯度下降法
    通过单独实现进行最优化的类，功能的模块化变得更简单
    """

    """
    SGD 的局限性在于
    对于有些函数，其形状可能是非均向的，这会造成其梯度方向并没有指向最小值的方向
    此时可能会使得 SGD 的搜索路径非常低效(比如呈之字形)
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        """
        将根据梯度更新参数的操作包装成一个类的实例方法
        """
        for key in params.keys():
            # 这里其实是一次迭代更新一层的(权重或偏置)参数
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    为克服 SGD 的局限性，有效应对非均向函数而提出的动量方法
    在力的作用下，物体的速度会逐渐增加
    使用此优化器更新(多元函数的)参数的时候，会在本参数分量的方向上有一定的加速度
    所以可以在一定程度上避免低效的搜索路径(比如之字形)
    """
    def __init__(self, lr=0.01, momentum=0.9):
        # 这里的 momentum(动量)是个关键参数，通常取 0.9
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                # 速度全部初始化为 0
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # 此方法本质上还是沿着负梯度的方向逐步更新参数，只是形式上跟上面的 SGD 有很大不同
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """
    AdaGrad(AdaptiveGradient) 是一种针对每个参数应用学习率衰减的思路
    即随着学习的进行，使学习率逐渐减小(一开始“多”学，然后逐渐“少”学)
    而且针对每个参数，赋予其【定制】的值
    它保存了以前的所有梯度值的平方和(是一个与参数矩阵同型的矩阵),可以按参数的元素进行学习率衰减，使变动大的参数的学习率逐渐减小
    此优化器也可有效客克服 SGD 的局限性
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                # 一开始先将所有梯度值的平方初始化为 0
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            # 下面的代码相当于学习率乘以 1/h，最后面加的那个微小值是为了防止 0 做除数的情况发生
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    """
    Adam (http://arxiv.org/abs/1412.6980v8)
    此优化器的思路书上并未做详细介绍，只说是相当于将 Momentum 和 AdaGrad 的思路融合到了一起，详情参考上面的论文
    这里先只给出代码而不做详细讨论，后面有机会再详细分析其算法思路
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
