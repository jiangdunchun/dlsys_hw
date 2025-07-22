"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True, device=device, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        h = X @ self.weight

        if not self.bias:
            return h
            
        return h + self.bias.broadcast_to(h.shape)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        assert len(X.shape) > 1

        shape = list(X.shape)
        batch_size, num_features = shape[0], 1
        for axis in shape[1:]:
            num_features *= axis

        return X.reshape((batch_size, num_features))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        H_i, H_o = None, x
        for module in self.modules:
            H_i = H_o
            H_o = module.forward(H_i)
        return H_o
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        assert len(logits.shape) == 2

        batch_size, num_features = logits.shape[0], logits.shape[1]

        y_one_hot = init.one_hot(num_features, y)

        e_Z = ops.exp(logits)
        sum_Z = e_Z.sum(1).reshape((batch_size, 1))
        e_Z = e_Z / sum_Z.broadcast_to(e_Z.shape)

        losses = -1. * ops.log((e_Z * y_one_hot).sum(1))

        #@TODO: float32 / int = float64
        return losses.sum() / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) == 2
        
        batch_size, num_features = x.shape[0], x.shape[1]

        if self.training:
            E_x = x.sum(0).reshape((1, num_features)) / batch_size
            Var_x = ((x - E_x.broadcast_to(x.shape)) ** 2).sum(0).reshape((1, num_features)) / batch_size

            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * E_x.reshape(num_features)
            self.running_var = (1. - self.momentum) * self.running_var + self.momentum * Var_x.reshape(num_features)
        else:
            E_x, Var_x = self.running_mean.reshape((1, num_features)), self.running_var.reshape((1, num_features))

        x_normalized = (x - E_x.broadcast_to(x.shape)) / ((Var_x.broadcast_to(x.shape) + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * x_normalized + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) == 2

        batch_size, num_features = x.shape[0], x.shape[1]

        E_x = x.sum(1).reshape((batch_size, 1)) / num_features
        Var_x = ((x - E_x.broadcast_to(x.shape)) ** 2).sum(1).reshape((batch_size, 1)) / num_features

        x_normalized = (x - E_x.broadcast_to(x.shape)) / ((Var_x.broadcast_to(x.shape) + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * x_normalized + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training: return x

        mask = init.randb(*x.shape, p=(1. - self.p))
        return x * mask / (1. - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        F_x = self.fn.forward(x)

        assert F_x.shape == x.shape
        
        return F_x + x
        ### END YOUR SOLUTION
