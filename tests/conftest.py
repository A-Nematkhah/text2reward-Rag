import os
import sys
from pathlib import Path


def pytest_configure(config):
    """Ensure project root is on sys.path for tests run by pytest/CI."""
    root = Path(__file__).resolve().parents[1]
    root_path = str(root)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
