"""PPO training loop, logging, and visualization."""

from txt2reward.training.callbacks import RewardEvolutionCallback
from txt2reward.training.device import detect_device
from txt2reward.training.env_factory import build_vec_env, make_env
from txt2reward.training.logger import TrainingLogger
from txt2reward.training.plots import generate_all_plots

__all__ = [
    "TrainingLogger",
    "generate_all_plots",
    "RewardEvolutionCallback",
    "build_vec_env",
    "make_env",
    "detect_device",
]
