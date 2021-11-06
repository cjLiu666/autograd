from os import error
import numpy as np
from numpy.core.fromnumeric import shape
from autograd.tensor import Tensor
from autograd.function import relu

x_data = Tensor(np.random.randn(100, 10))
coef1 = Tensor(np.ones((10, 3), dtype=float))
bias1 = Tensor(5.0)
coef2 = Tensor(np.array([1, 2, 3], dtype=float))
bias2 = Tensor(2.0)
y_data = (x_data @ coef1 + bias1)
print(y_data, coef1, coef2, bias1, bias2)


w1 = Tensor(np.random.randn(10, 3), requires_grad=True)
b1 = Tensor(np.random.randn(), requires_grad=True)
w2 = Tensor(np.random.randn(3), requires_grad=True)
b2 = Tensor(np.random.randn(), requires_grad=True)

print(w1, w2, b1, b2)
learning_rate = 0.001
batch_size = 32

for epoch in range(100):
    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = min(start + batch_size, 100)

        w1.zero_grad()
        b1.zero_grad()

        # w2.zero_grad()
        # b2.zero_grad()

        input = x_data[start:end]

        predicted = input @ w1 + b1
        # x2 = relu(x1)
        # predicted =  x2 @ w2 + b2
        print(predicted)

        actual = y_data[start:end]
        errors = predicted - actual
        loss = (errors * errors).sum()
        print(loss)

        loss.backward()
        epoch_loss += loss.data

        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad

        # w2 -= learning_rate * w2.grad
        # b2 -= learning_rate * b2.grad
        

    print(epoch, epoch_loss)
    print(w1, b1)


