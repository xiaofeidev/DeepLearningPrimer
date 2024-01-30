import sys, os
from collections import OrderedDict

from layer.layer import Affine, Relu, SoftmaxWithLoss

sys.path.append(os.pardir)
from neural_network.activation import *
from gradient.gradient import numerical_gradient
from num_recog.data.mnist import load_mnist
import math
import time


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        参数从头开始依次表示输入层的神经元数、隐藏层的神经元数(注意只有一个隐藏层)、输出层的神经元数、参数标准差的缩放因子
        注意 np.random.randn 函数默认会生成均值（正态分布的中心点）为 0，标准差为 1 的数值的矩阵，这里把标准差定为 0.01，
        则初始化出来的权重参数的值的取值范围会大概率分布在 [-0.1, 0.1] 的范围内，且均值仍然为 0，注意前面的表述：【大概率】
        """
        # 初始化权重和偏置，用一部字典存放
        self.params = {}
        # 输入层和隐藏层之间的权重参数
        # 权重使用符合高斯分布的随机数进行初始化，是个二维数组
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 输入层和隐藏层之间的偏置参数
        # 偏置使用0进行初始化，是个一维数组
        self.params['b1'] = np.zeros(hidden_size)

        # 隐藏层和输出层之间的权重参数
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 隐藏层和输出层之间的偏置参数
        self.params['b2'] = np.zeros(output_size)

        #  至此，各层的权重和偏置已初始化好
        ########
        #  用【有序字典】组织各个计算结点层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        ########

    def predict(self, x):
        """
        神经网络的推理（预测）函数，输入 x 为（二维像素值展开的）一维数组
        注意现在的向前推理函数，没有输出层激活函数(这里是 softmax)的参与
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        计算预测的损失函数，交叉熵误差
        x:用于神经网络推理的输入，预期为二维数组, t:监督数据，预期为二维数组
        """
        y = self.predict(x)
        #  计算误差的时候就需要输出层激活函数(以及损失函数)的参与了
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """
        预测的准确度
        x:输入，预期为二维数组, t:监督数据，预期为二维数组
        """
        y = self.predict(x)  # 推理结果

        y = np.argmax(y, axis=1)

        if t.ndim != 1:  # 不是 1 就是 2
            #  如果监督数据是 one-hot 形式，则做此处理
            #  否则不需要处理
            t = np.argmax(t, axis=1)
        # 保留四位小数
        accuracy = round(np.sum(y == t) / float(x.shape[0]), 4)
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            # 反向传播计算出各个计算层中各参数的偏导数
            dout = layer.backward(dout)
        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

    def num_gradient(self, x, t):
        """
        神经网络的参数的（数值微分）梯度的计算
        x 是用于神经网络推理的输入数据
        t 是标签信息
        """
        # 也即求损失函数对各个参数的偏导数，因为推理结果 y 是所有参数(和输入)的函数，而损失函数又是关于 x 的函数
        # 所以损失函数其实是关于所有参数的（复合）函数
        # 【注意这里计算的其实是多条数据的平均交叉熵误差】
        loss_W = lambda W: self.loss(x, t)

        grads = {'W1': numerical_gradient(loss_W, self.params['W1']),
                 'b1': numerical_gradient(loss_W, self.params['b1']),
                 'W2': numerical_gradient(loss_W, self.params['W2']),
                 'b2': numerical_gradient(loss_W, self.params['b2'])}
        return grads


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:.2f}"


if __name__ == "__main__":
    """
    下面开始讲训练数据分批来训练神经网络(的参数)，用数值微分梯度
    用于在 MNIST 数据集上进行手写数字识别
    其实直接运行下面的代码(用 CPU 训练，低效的数值微分计算梯度算法)会非常非常的慢，待以后看看 BP 算法的版本运行效率如何
    """
    start_time = time.time()

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    # 超参数
    iters_num = 9600  # 参数更新次数，这里共对参数进行 9600 次更新，每一次都将所有参数的值更新一遍(共 16 个 epoch)
    train_size = x_train.shape[0]  # 60000，共包含 60000 条训练数据
    batch_size = 100  # 每个数据批次包含 100 条数据(这里是 100 张图片)
    learning_rate = 0.1  # 学习率，是否有点偏大？

    train_loss_list = []  # 用于存放每次更新参数后的损失函数值，以记录学习轨迹

    # 训练的神经网络的参数在训练数据上的推理精度，每个epoch统计一次
    train_acc_list = []
    # 训练的神经网络的参数在测试数据上的推理精度，每个epoch统计一次
    test_acc_list = []
    # 平均每个epoch的重复次数
    #  epoch是一个单位，一个epoch表示学习中所有训练数据均被使用过一次时的更新次数
    iter_per_epoch = math.floor(max(train_size / batch_size, 1))

    count = 0
    # 初始化一个两层神经网络
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    for i in range(iters_num):
        # 获取 mini-batch
        # 从 [0, 60000) 中随机挑选 100 个数字，作为一个大小为 100 的批数据的下标
        # 这里的 mini-batch 每次都是随机选择的，所以参数更新完 iters_num 次都不一定每个数据都会被看到
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]  # 含 100 条数据
        t_batch = t_train[batch_mask]  # 含 100 条数据
        # 计算梯度
        # grad = network.num_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)  # 高速版!
        # 更新参数，每一次都根据神经网络的梯度将所有的参数都更新一次
        for key in ('W1', 'b1', 'W2', 'b2'):
            # 梯度下降法更新参数的值
            network.params[key] -= learning_rate * grad[key]
        # 记录学习过程，正常情况下，随着学习的进行，损失函数的值会不断减小，这是学习正常进行的信号
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        count += 1
        print(f'{count} rounds of parameter updates completed')
        # 计算每个 epoch 的识别精度，对此处而言，最外层的 for 循环执行 600 次即为一个 epoch
        if count % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    end_time = time.time()
    execution_time = end_time - start_time
    formatted_time = format_time(execution_time)
    print(f"Training time: {formatted_time}")

    """
    将上面训练出来的参数作为文件保存起来
    """
    import pickle
    import datetime

    # 获取当前的日期和时间
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # 构建pickle文件名，例如 "model_20231026_165812.pkl"
    pickle_filename = f"md_{timestamp}.pkl"

    # 使用pickle模块将字典保存到 pickle 文件
    with open(pickle_filename, 'wb') as file:
        # 下面的第一个参数传入要序列化保存的字典
        pickle.dump(network.params, file)

    # print(f"字典已保存为 {pickle_filename} 文件。")
    print(f"模型文件保存至源码同目录。")
