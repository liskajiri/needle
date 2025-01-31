from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from needle.nn.core import Parameter
from needle.optim.base import Optimizer

if TYPE_CHECKING:
    from collections.abc import Iterable


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
        self.weight_decay = weight_decay

        self.state = {}
        for p in self.params:
            self.state[p] = Parameter(np.zeros(p.data.shape))

    def step(self) -> None:
        for p in self.params:
            grad = p.grad.data + self.weight_decay * p.data
            curr_state = self.state[p]

            curr_state.data = (
                self.momentum * curr_state.data + (1 - self.momentum) * grad.data
            )
            p.data -= self.lr * curr_state.data
