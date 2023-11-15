"""
各种损失函数的代码实现
在进行神经网络的学习时，不能将识别精度作为指标。因为如果以
识别精度为指标，则参数的导数在绝大多数地方都会变为0。
"""
import numpy as np


def mean_squared_error(y, t):
    """
    均方误差的代码实现
    y 表示神经网络的输出，t 表示监督数据
    注意从手写数字识别例子的角度出发来考虑这边的两个参数
    这里的 y 和 t 应该都是长度为 10 的一维数组
    y 代表对一张图片输入的推理结果，数组的十个分量分别代表数字 0 - 9 的概率(y 是 softmax 函数的输出)
    t 是监督数据，采用 one-hot 表示，将正确解标签表示为1，其他标签表示为0
    """
    0.5 * np.sum((y - t) ** 2)


# def cross_entropy_error(y, t):
#     """
#     交叉熵误差的代码实现
#     交叉熵误差只计算对应正确解标签的输出(概率)的自然对数
#     """
#     delta = 1e-7 # 加上这个微小值是为了防止出现 np.log(0) 变为负无限大
#     return -np.sum(t * np.log(y + delta))

def cross_entropy_error(y, t):
    """
    交叉熵误差的代码实现
    可以同时处理单个数据和批量数据（数据作为batch集中输入）两种情况的函数
    【注意这里计算的其实是多条数据的平均交叉熵误差】
    """
    # y 的维度如果为 1，则调整其形状为简单的二维数组，只不过得到的这个二维数组只有一行
    # 二维数组的一行即为一条推理结果
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]  # 共有多少条推理结果数据
    # 注意下面的 sum 方法的实参如果是个二维数组，则会把二维数组中的所有数字相加成一个单独数字
    # 最后算出的是一个【平均交叉熵误差】
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error1(y, t):
    """
    当监督数据以标签形式给出，而非 one-hot 形式时
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    # 注意此时 np.arange(batch_size) 是一个由[0, batch_size) 组成的一维数组，t 也是个一维数组
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
