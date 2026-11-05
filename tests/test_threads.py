"""Thread defaults must be established before numerical libraries load in workers."""

import json
import os
import subprocess
import sys

import pytest

VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


@pytest.mark.parametrize("explicit", [None, "3"])
def test_package_thread_defaults_in_fresh_process(explicit):
    env = os.environ.copy()
    for key in VARIABLES:
        env.pop(key, None)
    if explicit is not None:
        env[VARIABLES[0]] = explicit
    source = (
        "import nbv, os, sys, json; "
        "print(json.dumps([os.environ.get(k) for k in " + repr(VARIABLES) + "])); "
        "assert 'numpy' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [explicit or "1", "1", "1"]
