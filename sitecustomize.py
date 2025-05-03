import os

# Only import and start coverage when COVERAGE_PROCESS_START is set
if os.environ.get("COVERAGE_PROCESS_START"):
    try:
        import coverage

        coverage.process_startup()
    except ImportError:
        pass
