# ---------------------------------------------------------------------------
# Shared pytest fixtures: the project config and the prepared data, built
# once per test session (the data is small, but there is no reason to
# reload and resplit it for every test).
# ---------------------------------------------------------------------------
"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from bcd.config import Config, load_config
from bcd.pipeline.experiment import PreparedData, prepare_data


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The real project configuration."""
    return load_config()


@pytest.fixture(scope="session")
def data(cfg: Config) -> PreparedData:
    """Data prepared exactly as the pipeline does it."""
    return prepare_data(cfg)
