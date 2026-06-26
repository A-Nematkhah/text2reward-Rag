"""PPO training and reward-evolution schedule defaults.

Tunable hyperparameters for ``train.py`` CLI defaults and designer wiring.
Constants only — no side effects on import.
"""

from __future__ import annotations

# Reward program hot-reload cadence (env steps per worker).
DEFAULT_RELOAD_INTERVAL = 200

# Evolution: episodes before the first LLM reward generation.
DEFAULT_WARMUP_EPISODES = 40

# Evolution: generate a new reward every N completed episodes (after warmup).
DEFAULT_EVOLVE_EVERY = 20

# Natural-language goal sent to the LLM on bootstrap / evolution.
DEFAULT_DRIVING_GOAL = (
    "Drive fast and safely on a 4-lane highway. Overtake slow vehicles. "
    "Avoid collisions. Prefer speeds above 25 m/s. Minimise harsh braking."
)

# PPO training CLI defaults.
DEFAULT_TOTAL_TIMESTEPS = 200_000
DEFAULT_N_ENVS = 4
DEFAULT_CHECKPOINT_FREQ = 10_000
DEFAULT_PLOT_SMOOTH_WINDOW = 10
DEFAULT_PLOT_DIR = "plots"

# PPO hyperparameters (stable-baselines3).
PPO_N_STEPS = 512
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 5
