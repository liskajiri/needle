from collections import defaultdict
from typing import Iterable
from needle.nn.nn_basic import Parameter
from needle.optim.base import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
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
            self.u[p] = self.beta1 * self.u[p] + (1 - self.beta1) * grad.data
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * grad.data**2

            # bias corrections
            u_hat = self.u[p] / (1 - self.beta1 ** (self.t))
            v_hat = self.v[p] / (1 - self.beta2 ** (self.t))

            # resolve issues with wrong types
            p.data -= self.lr * u_hat / ((v_hat) ** 0.5 + self.eps)
