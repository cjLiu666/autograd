import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union


Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Tensorable = Union['Tensor', float, np.ndarray]

def ensure_tensor(tensorable: Tensorable) ->'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if requires_grad:
            self.zero_grad()
        
    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        self.grad = None

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        return _add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        self.data = self.data + ensure_tensor(other).data
        return self

    def __sub__(self, other) -> 'Tensor':
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return _sub(ensure_tensor(other), self)

    def __isub__(self, other) -> 'Tensor':
        self.data = self.data - ensure_tensor(other).data
        return self

    def __mul__(self, other) -> 'Tensor':
        return _mul(self, ensure_tensor(other))
    
    def __rmul__(self, other) -> 'Tensor':
        return _mul(ensure_tensor(other), self)

    def __imul__(self, other) -> 'Tensor':
        self.data = self.data * ensure_tensor(other).data
        return self
    
    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def __matmul__(self, other) -> 'Tensor':
        return _matmul(self, other)

    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data += grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))


    def sum(self) -> 'Tensor':
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
        
    return Tensor(data, requires_grad, depends_on)


def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad  = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad  = grad.sum(axis=i, keepdims=True)
        
            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data

            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad  = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data

            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad  = grad.sum(axis=i, keepdims=True)
        
            return grad
        depends_on.append(Dependency(t2, grad_fn2))
    return Tensor(data, requires_grad, depends_on)


def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x : -x)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return _add(t1, _neg(t2))



def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def _slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad
    depends_on: List[Dependency] = []
    
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(t.data)
            bigger_grad[idxs] = grad
            return bigger_grad

        depends_on.append(Dependency(t, grad_fn))


    return Tensor(data, requires_grad, depends_on)


if __name__ == '__main__':

    # t1 = Tensor([1, 2, 3], requires_grad=True)
    # t2 = t1.sum()

    # t2.backward()
    # assert t1.grad.data.tolist() == [1, 1, 1]
    
    # t2.backward(grad=Tensor(3))
    # assert t1.grad.data.tolist() == [3, 3, 3]


    # t1 = Tensor([1, 2, 3], requires_grad=True)
    # t2 = Tensor([4, 5, 6], requires_grad=True)

    # t3 = add(t1, t2)
    # t3.backward(Tensor([-1, -2, -3]))

    # assert t1.grad.data.tolist() == [-1, -2, -3]
    # assert t2.grad.data.tolist() == [-1, -2, -3]

    # t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)  # (2, 3)
    # t2 = Tensor([7, 8, 9], requires_grad = True)               # (3,)

    # t3 = add(t1, t2)   # shape (2, 3)
    # t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

    # assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
    # assert t2.grad.data.tolist() == [2, 2, 2]

    # t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)    # (2, 3)
    # t2 = Tensor([[7, 8, 9]], requires_grad = True)               # (1, 3)

    # t3 = add(t1, t2)
    # t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

    # assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
    # assert t2.grad.data.tolist() == [[2, 2, 2]]

    # t1 = Tensor([1, 2, 3], requires_grad=True)
    # t2 = Tensor([4, 5, 6], requires_grad=True)

    # t3 = mul(t1, t2)
    # t3.backward(Tensor([-1., -2., -3.]))

    # assert t1.grad.data.tolist() == [-4, -10, -18]
    # assert t2.grad.data.tolist() == [-1,  -4,  -9]

    # t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)  # (2, 3)
    # t2 = Tensor([7, 8, 9], requires_grad = True)               # (3,)

    # t3 = mul(t1, t2)   # shape (2, 3)
    # t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

    # assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
    # assert t2.grad.data.tolist() == [5, 7, 9]

    # t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)    # (2, 3)
    # t2 = Tensor([[7, 8, 9]], requires_grad = True)               # (1, 3)

    # t3 = mul(t1, t2)
    # t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

    # assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
    # assert t2.grad.data.tolist() == [[5, 7, 9]]


    # t1 = Tensor([1, 2, 3], requires_grad=True)
    # t2 = Tensor([4, 5, 6], requires_grad=True)

    # t3 = sub(t1, t2)
    # t3.backward(Tensor([-1., -2., -3.]))

    # assert t1.grad.data.tolist() == [-1, -2, -3]
    # assert t2.grad.data.tolist() == [+1, +2, +3]

    # t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)  # (2, 3)
    # t2 = Tensor([7, 8, 9], requires_grad = True)               # (3,)

    # t3 = sub(t1, t2)   # shape (2, 3)
    # t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

    # assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
    # assert t2.grad.data.tolist() == [-2, -2, -2]

    t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)    # (2, 3)
    t2 = Tensor([[7, 8, 9]], requires_grad = True)               # (1, 3)

    t3 = t1 - t2
    t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

    assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert t2.grad.data.tolist() == [[-2, -2, -2]]



