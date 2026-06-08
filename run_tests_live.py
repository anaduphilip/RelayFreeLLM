# run_tests_live.py — live integration test suite
#
# NOTE: tests/manual/ scripts are NOT auto-discovered by pytest.
# They are standalone scripts run via:
#   python tests/manual/test_models_availability.py
#   python tests/manual/test_session_live.py
#   python tests/manual/test_context_live.py
#   etc.
#
# These require a running RelayFreeLLM server and/or real API keys.
#
# tests/performance/test_stability.py also requires a running
# server but is currently blocked by a FastAPI/Starlette version
# incompatibility (APIRouter.on_startup removed in Starlette >= 1.0).

import sys
import pytest

answer = input("Is the local RelayFreeLLM server running? (y/N): ").strip().lower()
if answer != "y":
    print("Aborted. Start the server first, then re-run.")
    sys.exit(0)

args = [
    "-v",
]

sys.exit(pytest.main(args))
