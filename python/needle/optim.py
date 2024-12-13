"""Optimization module"""

from collections import defaultdict
from typing import Iterable

from needle.nn import Parameter


class Optimizer:
    def __init__(self, params) -> None:
        self.params = params

    def step(self) -> None:
        raise NotImplementedError()

    def reset_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def zero_grad(self) -> None:
        # method to mimic Pytorch's syntax
        self.reset_grad()


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(lambda: 0.0)
        self.weight_decay = weight_decay

    def step(self) -> None:
        for p in self.params:
            grad = p.grad.data + self.weight_decay * p.data
            self.u[p] = self.momentum * self.u[p] + (1 - self.momentum) * grad.data
            # resolve issues with wrong types
            p.data -= self.lr * self.u[p]


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = defaultdict(lambda: 0.0)
        self.v = defaultdict(lambda: 0.0)

    def step(self) -> None:
        self.t += 1
        for p in self.params:
            grad = p.grad.data + self.weight_decay * p.data
            self.u[p] = self.beta1 * self.u[p] + (1 - self.beta1) * grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * grad**2

            # bias corrections
            u_hat = self.u[p] / (1 - self.beta1 ** (self.t))
            v_hat = self.v[p] / (1 - self.beta2 ** (self.t))

            # resolve issues with wrong types
            p.data -= self.lr * u_hat / ((v_hat) ** 0.5 + self.eps)
