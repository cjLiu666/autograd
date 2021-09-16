from os import error
import numpy as np
from autograd.tensor import Tensor

x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, 3, -2], dtype=float))
y_data = x_data @ coef + 5

w = Tensor(np.random.randn(3), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)

learning_rate = 0.001
batch_size = 32

for epoch in range(100):
    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = min(start + batch_size, 100)

        w.zero_grad()
        b.zero_grad()

        input = x_data[start:end]

        predicted = input @ w + b
        actual = y_data[start:end]
        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss.data

        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    print(epoch, epoch_loss)
    print(w, b)


