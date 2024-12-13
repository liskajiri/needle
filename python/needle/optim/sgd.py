from collections import defaultdict
from typing import Iterable
from needle.nn.nn_basic import Parameter
from needle.optim.base import Optimizer


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
