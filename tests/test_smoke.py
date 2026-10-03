"""Smoke tests: every module imports and config is sane."""

import importlib

import pytest

from mixlab import config

MODULES = [
    "data_gen",
    "validate",
    "transforms",
    "model",
    "evaluate",
    "insights",
    "optimizer",
    "ai_explainer",
    "config",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(f"mixlab.{name}").__doc__


def test_config_paths_inside_project() -> None:
    assert config.RAW_DIR.is_relative_to(config.PROJECT_ROOT)
    assert config.FIGURES_DIR.is_relative_to(config.PROJECT_ROOT)
