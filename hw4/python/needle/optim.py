"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for para in self.params:
            grad = para.grad.detach() + self.weight_decay * para.detach()

            if para not in self.u: self.u[para] = 0.
            u = self.u[para]
            u_1 = self.momentum * u + (1. - self.momentum) * grad
            self.u[para] = u_1

            #@TODO: the grad is float64 while the para is float32
            para.data = para.detach() - self.lr * ndl.Tensor(u_1, dtype=para.dtype)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for para in self.params:
            grad = para.grad.detach() + self.weight_decay * para.detach()

            if para not in self.m: 
                self.m[para], self.v[para] = 0., 0.

            self.m[para] = self.beta1 * self.m[para] + (1. - self.beta1) * grad
            self.v[para] = self.beta2 * self.v[para] + (1. - self.beta2) * (grad ** 2)

            m_hat, v_hat = self.m[para] / (1. - (self.beta1 ** self.t)), self.v[para] / (1. - (self.beta2 ** self.t))

            para.data = para.detach() - self.lr * ndl.Tensor(m_hat / ((v_hat ** 0.5) + self.eps), dtype=para.dtype)
            para.detach()
        ### END YOUR SOLUTION
