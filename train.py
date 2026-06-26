"""CLI entry point for PPO training with Text-to-Reward evolution."""

from txt2reward.training.train import *  # noqa: F403
from txt2reward.training.train import main

if __name__ == "__main__":
    main()
