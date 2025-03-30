from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from needle.backend_selection import array_api
from needle.nn.core import Parameter
from needle.optim.base import Optimizer

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(slots=True)
class AdamState:
    u: Parameter
    v: Parameter


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

        self.state = {}
        for p in self.params:
            zeros = array_api.zeros(p.data.shape, dtype=p.data.dtype)
            self.state[p] = AdamState(
                Parameter(zeros),
                Parameter(zeros),
            )

    def step(self) -> None:
        self.t += 1
        for p in self.params:
            grad = p.grad.data + self.weight_decay * p.data
            curr_state = self.state[p]

            curr_state.u.data = (
                self.beta1 * curr_state.u.data + (1 - self.beta1) * grad.data
            )
            curr_state.v.data = (
                self.beta2 * curr_state.v.data + (1 - self.beta2) * grad.data**2
            )

            # bias corrections
            u_hat = curr_state.u / (1 - self.beta1 ** (self.t))
            v_hat = curr_state.v / (1 - self.beta2 ** (self.t))

            p.data -= self.lr * u_hat / ((v_hat) ** 0.5 + self.eps)
