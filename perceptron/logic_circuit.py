# 逻辑电路 AND，注意输入 x1 和 x2 的取值只能是 0 或 1（下面其他逻辑电路函数同样）
# 输出也只能是 0 或 1
# 这里 w1, w2, theta 这三个参数是人为指定的，而非通过数据学习而得
# x1*w1 + x2*w2 > theta 则返回 1，否则返回 0
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
import numpy as np
# 逻辑电路 AND(和)，使用权重和偏置的实现
# 偏置用 b 表示
# x1*w1 + x2*w2 + b 如果大于0则输出1，否则输出0
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# 逻辑电路 NAND(与非)，也就是 AND 的否定
# 同样的输入，当 AND 为 1 时 NAND 为 0，当 AND 为 0 时，NAND 为 1
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) #仅权重和偏置与AND不同！
    b = 0.7 #仅权重和偏置与AND不同！
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# 逻辑电路 OR(或)
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) #仅权重和偏置与AND不同！
    b = -0.2 #仅权重和偏置与AND不同！
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# 感知机不能直接表示异或门(单层感知机无法表示异或门 or 单层感知机无法分离非线性空间)
# 但可通过【叠加层】即组合已有门电路的方式来实现
# 其实异或门可以通过组合与非门和或门来实现，即
# NAND && OR
# 这其实是个多层(2层)感知机
# 单层感知机无法表示的东西，通过增加一层就可以解决
# 通过叠加层（加深层），感知机能进行更加灵活的表示
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

