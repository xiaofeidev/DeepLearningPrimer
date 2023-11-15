import sys, os
sys.path.append(os.pardir)
import numpy as np
from num_recog.data.mnist import load_mnist

# 读入训练数据和训练数据标签，测试数据和测试数据标签
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# print(x_train.shape) # (60000, 784)
# print(t_train.shape) # (60000, 10)

train_size = x_train.shape[0] # 60000
batch_size = 10
# 从 [0, 60000) 中随机选取十个数字，返回 NumPy 数组
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

