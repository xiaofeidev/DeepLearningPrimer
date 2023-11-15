"""
神经网络中各种计算结点层的实现，各层都兼顾【正向】和【反向】传播，反向传播用于计算各参数的偏导数
一般假定 forward()和 backward() 的参数是NumPy数组
反向传播依据的原理就是复合函数求导
"""

"""
用层进行模块化的实现在工程层面具有非常大优点
"""

import numpy as np

from loss.loss import cross_entropy_error
from neural_network.activation import softmax


class MulLayer:
    """
    乘法层
    """

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        """
        正向传播没啥好说的，只是简单执行对应的计算并返回
        注意两个参数更多是 NumPy 数组
        """
        self.x = x
        self.y = y
        out = x * y
        # 这里没有存储正向传播时本结点的输出值 out
        return out

    def backward(self, dout):
        """
        dout 是下游传播过来的导数，如果当前结点本就位于末端，则 dout 取 1
        下面的计算根据于求导乘法法则
        注意参数更多是 NumPy 数组
        """
        dx = dout * self.y  # 翻转x和y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    """
    加法层
    """

    def __init__(self):
        pass

    def forward(self, x, y):
        """
        正向传播
        注意两个参数更多是 NumPy 数组
        """
        out = x + y
        return out

    def backward(self, dout):
        """
        反向传播，只是把下游传过来的导数原封不动地传给上游
        dout 是下游传播过来的导数，如果当前结点本就位于末端，则 dout 取 1
        注意参数更多是 NumPy 数组
        """
        dx = dout * 1
        dy = dout * 1
        return dx, dy


"""
下面是常用激活函数的层实现
"""


class Relu:
    """
    ReLU(激活函数)层
    """

    def __init__(self):
        # mask 实例变量是一个由 True/False 构成的 NumPy 数组
        # 其会将正向传播时的输入 x 的元素中小于等于0的地方保存为 True，否则为 False
        # 此激活函数层在正向传播的时候会记录相应的中间值，以在反向传播时使用
        self.mask = None

    def forward(self, x):
        """
        正向传播，参数 x 是个一维 NumPy 数组
        """
        self.mask = (x <= 0)  # 初始化 mask 数组
        out = x.copy()
        out[self.mask] = 0  # out 数组中与 mask 数组中值为 True 的相同下标位置的值会被设为 0
        return out

    def backward(self, dout):
        """
        反向传播，参数 dout 是个 NumPy 数组
        """
        dout[self.mask] = 0  # out 数组中与 mask 数组中值为 True 的相同下标位置的值会被设为 0
        dx = dout
        return dx


class Sigmoid:
    """
    Sigmoid 激活函数层
    可以从数学上证明，Sigmoid 层的反向传播，只根据正向传播的输出就能计算出来
    """

    def __init__(self):
        # 这个实例变量用于存储正向传播时的输出值
        self.out = None

    def forward(self, x):
        """
        正向传播，x 是个数组
        需要存储正向传播时的输出值，以在反向传播时用来计算偏导数
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        """
        仅使用正向传播时的输出值便可计算出反向传播时的偏导数，利用即成的公式
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx


######################################################################
class Affine:
    """
    矩阵乘法层，矩阵乘法又称仿射变换，故得此类名
    """

    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        x 通常是个二维数组，其中每一行数据的长度(也就是列数)就是正向传播时上一层的结点数
        """
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)  # 有现成的矩阵求导公式
        self.dW = np.dot(self.x.T, dout)  # 有现成的矩阵求导公式
        self.db = np.sum(dout, axis=0)  # 偏置的偏导数要将下游传过来的偏导数矩阵的每一列求和，这是合理的
        return dx  # 返回 dx 是否正确？


class SoftmaxWithLoss:
    """
    Softmax 激活函数加(交叉熵误差)损失函数层
    """

    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # softmax的输出
        self.t = None  # 监督数据（one-hot vector）

    def forward(self, x, t):
        """
        x: 上游传来的的输出值，作为此处向前传播的输入值
        """
        self.t = t
        self.y = softmax(x)
        # 下面这行计算的其实是 batch_size 条数据的【平均】交叉熵误差
        self.loss = cross_entropy_error(self.y, self.t)
        # 这一层算是在下游的末端了，输出的直接是最后损失函数的值
        return self.loss

    def backward(self, dout=1):
        """
        反向传播
        注意此结点是反向传播的起始处，所以 dout 取 1
        """
        batch_size = self.t.shape[0]
        # 这里为什么要除以 batch_size? 似乎是为了使梯度的规模变小(也就是使梯度的值变小)，这可能有助于控制梯度爆炸或梯度消失的问题
        # 注意这里之所以要除以 batch_size，是因为在上面的正向传播里面，cross_entropy_error 中的计算也将最后得到的值除以了 batch_size
        # 而这在反向传播的导数计算看来，相当于是计算函数时候的一个常数因子(1/batch_size)，所以在反向传播时计算导数的时候也得再带上这个常数因子，因为求导数时是可以先把常数提出来的！
        dx = dout * (self.y - self.t) / batch_size
        return dx
