import subprocess
import sys


def test_needle_import(benchmark, module_name="needle") -> None:
    """Benchmark import time for the main needle package."""

    def import_in_subprocess():
        subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            check=True,
        )

    benchmark(import_in_subprocess)
