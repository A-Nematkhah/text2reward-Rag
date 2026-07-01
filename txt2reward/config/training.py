"""PPO training and reward-evolution schedule defaults.

Tunable hyperparameters for ``train.py`` CLI defaults and designer wiring.
Constants only — no side effects on import.
"""

from __future__ import annotations

# Reward program hot-reload cadence (env steps per worker).
DEFAULT_RELOAD_INTERVAL = 200

# Per-step shaped reward clip in LLMRewardWrapper — stabilises PPO value learning
# when LLM-generated penalties grow large (e.g. collision -100, stacked taxes).
REWARD_STEP_CLIP_MIN = -10.0
REWARD_STEP_CLIP_MAX = 10.0

# Evolution: episodes before the first LLM reward generation.
DEFAULT_WARMUP_EPISODES = 80

# Evolution: generate a new reward every N completed episodes (after warmup).
DEFAULT_EVOLVE_EVERY = 100

# Skip LLM archive/generation while window crash_rate is at or above this threshold.
# Lets PPO improve on a fixed reward before the LLM inflates penalties.
EVOLVE_MAX_CRASH_RATE = 0.70

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
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_ENT_COEF = 0.02
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

# VecNormalize: stabilises critic when shaped rewards have high return variance.
DEFAULT_VEC_NORMALIZE_REWARD = True
VEC_NORMALIZE_CLIP_REWARD = 10.0
VEC_NORMALIZE_STATS_PATH = "vec_normalize.pkl"
