import importlib
import sys


def test_needle_import(benchmark, module_name="needle") -> None:
    """Benchmark import time for the main needle package."""
    original_modules = dict(sys.modules)

    def import_module():
        sys.modules.clear()
        sys.modules.update(
            {
                k: v
                for k, v in original_modules.items()
                if k.startswith("_") or "." not in k
            }
        )
        return importlib.import_module(module_name)

    result = benchmark(import_module)
    assert result is not None
