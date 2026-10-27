import importlib

import pytest

MODULES = [
    "nbv",
    "nbv.parts",
    "nbv.candidates",
    "nbv.visibility",
    "nbv.robot",
    "nbv.env",
    "nbv.baselines",
    "nbv.evaluation",
    "nbv.agents",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_version():
    import nbv

    assert nbv.__version__
