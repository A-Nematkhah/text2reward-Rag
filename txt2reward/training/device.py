"""Hardware device detection for PPO."""

from __future__ import annotations


def detect_device() -> str:
    """Return ``cuda``, ``mps``, or ``cpu`` for stable-baselines3 PPO."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
