"""Optimization module."""

from collections.abc import Iterable

from needle.nn.nn_basic import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter]) -> None:
        self.params = params

    def step(self) -> None:
        raise NotImplementedError

    def reset_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def zero_grad(self) -> None:
        # method to mimic Pytorch's syntax
        self.reset_grad()
