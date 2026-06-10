"""Smoke tests: the live modules must at least import.

Run on a box with the real deps installed (CPU instance / conda env txt23d).
Each test skips cleanly where deps are missing so the suite stays green
locally.
"""
import importlib

import pytest

LIVE_MODULES = [
    "app.config",
    "app.models",
    "app.main",
    "app.gradio.web_app",
    "worker.lib.manual_gen_schema",
    "worker.lib.manual_gen_queue",
]


@pytest.mark.parametrize("module", LIVE_MODULES)
def test_live_module_imports(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as e:
        pytest.skip(f"dependency not installed locally: {e.name}")
