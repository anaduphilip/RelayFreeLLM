# run_tests.py — offline test suite (no live server needed)
#
# Excludes:
#   tests/manual/       — live integration tests (need real API keys / server)
#   tests/performance/  — stability benchmarks
#
# The following files are also excluded (FastAPI/Starlette compat issue
# with on_startup/on_shutdown in APIRouter, unrelated to this project):
#   tests/test_integration_routing.py
#   tests/test_streaming.py

import sys
import pytest

excluded = [
    "tests/manual",
    "tests/performance",
    "tests/test_integration_routing.py",
    "tests/test_streaming.py",
]

args = ["tests", "-v"]
for path in excluded:
    args.extend(["--ignore", path])

sys.exit(pytest.main(args))
