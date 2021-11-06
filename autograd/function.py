from numpy.core.fromnumeric import transpose
from numpy.testing._private.utils import requires_memory
from autograd.tensor import Dependency, Tensor
import numpy as np

def relu(t: Tensor) -> Tensor:
    data = t.data * (t.data > 0)
    requires_grad = t.requires_grad
    
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (t.data > 0)
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)