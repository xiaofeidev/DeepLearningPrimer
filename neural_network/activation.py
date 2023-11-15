# 各种激活函数的实现

# 神经网络的结构从左往右看分别包括【输入层】【中间层(隐藏层)】【输出层】
# 注意这里的【激活函数】的重要概念
# 激活函数是连接感知机和神经网络的桥梁
# 下面给出书中列举出的各种激活函数的 python 代码实现

def step_function(x):
    '''
    阶跃函数的简单实现，注意参数 x 只接受一个实数(浮点数)
    '''
    if x > 0:
        return 1
    else:
        return 0

# 阶跃函数接收参数为 NumPy 数组的实现
# 阶跃函数显然在 x=0 这一点不连续(跳跃间断点)
import numpy as np
def step_function(x): # 注意这里的参数 x 是一个 NumPy 数组
    y = x > 0
    return y.astype(np.int8)

'''
# 绘制阶跃函数的图形
import matplotlib.pylab as plt

# ↓ 在−5.0到5.0的范围内，以0.1为单位，生成 NumPy 数组
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定 y 轴的范围
plt.show()
'''


# 思考：为什么会有这么个激活函数
def sigmoid(x):
    '''
    sigmoid函数的实现
    这里的参数 x 即可以是一个实数，也可以是一个 NumPy 数组
    这依托于 NumPy 的广播功能
    '''
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    双曲正切函数，值域 (-1,1)，图像也是 S 形，且关于原点对称
    """
    return np.tanh(x)


# sigmoid函数是一条平滑的曲线，sigmoid函数的平滑性对神经网络的学习具有重要意义
# 感知机中神经元之间流动的是0或1的二元信号，而神经网络中流动的是连续的实数值信号
# 神经网络的激活函数必须使用非线性函数，激活函数不能使用线性函数

# ReLU 函数
# 在输入大于0时，直接输出该值；在输入小于等于0时，输出0
# 从代码可以看出来这里的参数 x 只能是一个数
# 思考：为什么会有这么个激活函数
def relu(x):
    return np.maximum(0, x)


# (一般用于输出层)特殊的激活函数：恒等函数
# 其会将输入按原样输出
def identity_function(x):
    return x


# (一般用于输出层) softmax 函数
# 其特点是：输出层的各个神经元都受到【所有】输入信号的影响
# 其会使输出层的所有神经元的值之和为 1，且所有的输出值均为[0.1,1.0]之间的实数
# 因此，将其用于输出层时可解释为概率
# 注意这里的参数 a 是一个 numpy 数组
def softmax_old(x):
    if x.ndim == 2:
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

    return np.exp(x) / np.sum(np.exp(x))


# softmax 函数的防计算溢出版本
# 可以证明，在进行softmax的指数函数的运算时
# 将每个输入值加上（或者减去）某个常数并不会改变运算的结果
def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)  # 溢出对策
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
