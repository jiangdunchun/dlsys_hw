from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=1, keepdims=True)
        logsumexp_Z = array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis=1, keepdims=True)) + max_Z
        return Z - logsumexp_Z 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        max_Z = Tensor(array_api.max(lhs.realize_cached_data(), axis=1, keepdims=True))
        exp_Z = exp(lhs - max_Z.broadcast_to(lhs.shape))
        softmax_Z = exp_Z / exp_Z.sum(1).reshape(max_Z.shape).broadcast_to(lhs.shape)
        return [out_grad - softmax_Z * out_grad.sum(1).reshape(max_Z.shape).broadcast_to(lhs.shape)]
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        ret = array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis=self.axes, keepdims=True)) + max_Z
        return array_api.squeeze(ret)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        max_Z = Tensor(array_api.max(lhs.realize_cached_data(), axis=self.axes, keepdims=True))
        exp_Z = exp(lhs - max_Z.broadcast_to(lhs.shape))
        grad = exp_Z / exp_Z.sum(self.axes).reshape(max_Z.shape).broadcast_to(lhs.shape)
        return [out_grad.reshape(max_Z.shape).broadcast_to(lhs.shape) * grad]
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

