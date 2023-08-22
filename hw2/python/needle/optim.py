"""Optimization module"""


from collections import defaultdict
from typing import Iterable

from needle.nn import Parameter
from needle.autograd import Tensor


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(lambda: 0.0)
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data
            self.u[param] = (
                self.momentum * self.u[param] + (1 - self.momentum) * grad.data
            )
            # resolve issues with wrong types
            param.data -= self.lr * Tensor(self.u[param], dtype="float32")


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
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

        self.u = defaultdict(lambda: 0.0)
        self.v = defaultdict(lambda: 0.0)

    def step(self):
        self.t += 1
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data
            self.u[param] = self.beta1 * self.u[param] + (1 - self.beta1) * grad.data
            self.v[param] = (
                self.beta2 * self.v[param] + (1 - self.beta2) * grad.data**2
            )

            # bias corrections
            u_hat = self.u[param] / (1 - self.beta1 ** (self.t))
            v_hat = self.v[param] / (1 - self.beta2 ** (self.t))

            # resolve issues with wrong types
            param.data -= Tensor(
                self.lr * u_hat / ((v_hat) ** 0.5 + self.eps),
                dtype="float32",
            )
