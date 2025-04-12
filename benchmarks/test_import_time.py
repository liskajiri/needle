import importlib
import sys

import pytest


@pytest.mark.benchmark(
    min_rounds=10,
    max_time=0.5,
    max_iterations=10,
    disable_gc=True,
    warmup=True,
    warmup_iterations=1,
)
def test_needle_import(benchmark, module_name="needle"):
    """Benchmark import time for the main needle package."""
    original_modules = dict(sys.modules)

    def import_module():
        sys.modules.clear()
        sys.modules.update({
            k: v
            for k, v in original_modules.items()
            if k.startswith("_") or "." not in k
        })
        return importlib.import_module(module_name)

    result = benchmark(import_module)
    assert result is not None
