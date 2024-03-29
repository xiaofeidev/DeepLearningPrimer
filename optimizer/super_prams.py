"""
超参数的验证
事先将数据集中的数据分成训练数据、验证数据、测试数据三部分
其中验证数据专门用于超参数的性能评估，注意对超参数进行性能评估绝不能使用测试数据进行
有一种实践性的超参数的最优化方法，一开始先大致设定一个范围，从这个范围中随机选出一个超参数（采样），
用这个采样到的值进行识别精度的评估；然后，多次重复该操作，观察识别精度的结果，根据这个结果缩小超参数的“好值”的范围。
通过重复这一操作，就可以逐渐确定超参数的合适范围。

如果需要更严密的方法，可以使用贝叶斯最优化
"""