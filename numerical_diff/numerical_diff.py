import numpy as np

def numerical_diff(f, x):
    """
    数值微分计算函数 f 在 x 处的导数的近似值
    注意舍入误差问题，增量不能太小
    采用中心差分
    """
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
