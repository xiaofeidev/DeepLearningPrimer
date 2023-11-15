import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
# temp = np.max(x, axis=1)
# print(temp)
# x = x - temp
print(x)

original_x_shape = x.shape
x = x.reshape(3, -1)
print(x)

class A:
    def fun(self):
        print("funA")

class B:
    def fun(self):
        print("funB")


class C(A, B):
    pass

C().fun()  #funA
