import os
import sys
from pathlib import Path


def pytest_configure(config):
    """Ensure project root is on sys.path for tests run by pytest/CI."""
    root = Path(__file__).resolve().parents[1]
    root_path = str(root)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    config.addinivalue_line(
        "markers",
        "evolution: Task 7 evolution-system tests (fitness, bank, archive, curriculum)",
    )


def pytest_collection_modifyitems(config, items):
    """Tag the consolidated Task 7 module for selective runs."""
    for item in items:
        if "test_evolution_system.py" in str(item.fspath):
            item.add_marker("evolution")
