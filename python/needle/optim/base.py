"""Optimization module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from needle.nn.core import Parameter

if TYPE_CHECKING:
    from collections.abc import Iterable


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, params: Iterable[Parameter]) -> None:
        self.params = params

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    def reset_grad(self) -> None:
        for p in self.params:
            p.grad = Parameter(p * 0)

    def zero_grad(self) -> None:
        # method to mimic Pytorch's syntax
        self.reset_grad()
