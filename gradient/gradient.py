import numpy as np


def numerical_gradient(f, x):
    """
    神经网络的(某层所有参数的)梯度的计算
    f：损失函数
    x：神经网络的某一层的参数，如为权重则是二维数组，如为偏置则为一维数组
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    #  flags=['multi_index'] 告诉迭代器要返回多维索引
    #  op_flags=['readwrite'] 允许在迭代中修改数组中的元素
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #  这样的实现不管参数 x 是几维数组都能够一个一个元素地迭代数组中的所有元素
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()  # 下一个元素

    return grad

################################################
#  注意下面定义的函数仅用来帮助我们理解数值微分的代码实现，实际没什么用
# def _numerical_gradient_1d(f, x):
#     """
#     参数 x 为【一维】数组的数值梯度计算
#     多元函数对全部变量的偏导数汇总而成的向量称为梯度
#     这里的参数 f 为损失函数(是一个多元函数)
#     注意这里的参数 x 是一个一维数组
#     """
#     h = 1e-4 # 0.0001
#     # 生成一个形状和 x 相同、所有元素都为0的数组
#     grad = np.zeros_like(x)
#     for i in range(x.size):
#         tmp_val = x[i]
#         # f(x+h)的计算
#         x[i] = tmp_val + h
#         fxh1 = f(x)
#         # f(x-h)的计算
#         x[i] = tmp_val - h
#         fxh2 = f(x)
#         grad[i] = (fxh1 - fxh2) / (2*h)
#         x[i] = tmp_val # 还原值
#     return grad
#
#
# def _numerical_gradient_2d(f, x):
#     """
#     参数 f 为损失函数
#     参数 x 为二维数组，每一行为多元函数 f 的一组自变量，注意这个函数多我们应该没什么用
#     二维数组(某层的权重参数矩阵)的数值梯度计算
#     """
#     if x.ndim == 1:
#         return _numerical_gradient_1d(f, x)
#     else:
#         grad = np.zeros_like(x)
#
#         for i, x in enumerate(x):
#             grad[i] = _numerical_gradient_1d(f, x)
#
#         return grad
#
#
# def gradient_descent(f, init_x, lr=0.01, step_num=100):
#     """
#     梯度下降法的极简实现，这里只是机械地重复 100 次参数更新
#     这里的学习率是 0.01，学习率这个值过大或者过小都不好
#     学习率这样的参数称为超参数
#     """
#     x = init_x
#     for i in range(step_num):
#         grad = _numerical_gradient_1d(f, x)
#         x -= lr * grad
#     return x

