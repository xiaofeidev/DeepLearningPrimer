# 书中提供的下载和缓存 MNIST 数据集的脚本
'''
Python有pickle这个便利的功能。这个功能可以将程序运行中的对
象保存为文件。如果加载保存过的pickle文件，可以立刻复原之前
程序运行中的对象。用于读入MNIST数据集的 load_mnist() 函数内
部也使用了pickle功能（在第2次及以后读入时）
'''
# coding: utf-8
import numpy as np
# import sys, os
# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 输出各个数据的形状
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)

# 下面的代码为顺便显示 MNIST 数据集中的第一张图像
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)