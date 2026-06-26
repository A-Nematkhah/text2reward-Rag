"""Shared pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.helpers import passing_reward_code  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: end-to-end pipeline tests (metrics → archive → evolution)",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "test_evolution_system.py" in str(item.fspath):
            item.add_marker("integration")


@pytest.fixture
def passing_reward() -> str:
    return passing_reward_code()
