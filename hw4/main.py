import os
os.environ["PYTHONPATH"] = "/data/coding/dlsys_hw/hw4/python"
os.environ["NEEDLE_BACKEND"] = "nd"

import sys
sys.path.append('/data/coding/dlsys_hw/hw4/python')
import itertools
import numpy as np
import mugrade
import torch

import needle as ndl
from needle import backend_ndarray as nd

def test_stack(shape, axis, l, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [ndl.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
    A_t = [torch.Tensor(_A[i]) for i in range(l)]
    out = ndl.stack(A, axis=axis)
    out_t = torch.stack(A_t, dim=axis)
    np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)

def test_stack_backward(shape, axis, l, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [ndl.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
    A_t = [torch.Tensor(_A[i]) for i in range(l)]
    for i in range(l):
        A_t[i].requires_grad = True
    ndl.stack(A, axis=axis).sum().backward()
    torch.stack(A_t, dim=axis).sum().backward()
    for i in range(l):
        np.testing.assert_allclose(A_t[i].grad.numpy(), A[i].grad.numpy(), atol=1e-5, rtol=1e-5)

test_stack_backward((5, 5), 0, 1, ndl.cpu())