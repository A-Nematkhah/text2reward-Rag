"""
PPO training orchestration for highway-v0 with Text-to-Reward evolution.

Business logic lives in submodules; this file wires CLI → env → PPO → callbacks.
"""

from __future__ import annotations

import os

import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Re-export for tests that import ENV_CONFIG from train
from txt2reward.config.env import ENV_CONFIG  # noqa: F401
from txt2reward.config.paths import (
    ARCHIVE_FILE,
    LOG_FILE,
    REWARD_PROGRAM_PATH,
)
from txt2reward.config.training import (
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_DRIVING_GOAL,
    DEFAULT_EVOLVE_EVERY,
    DEFAULT_N_ENVS,
    DEFAULT_PLOT_DIR,
    DEFAULT_PLOT_SMOOTH_WINDOW,
    DEFAULT_RELOAD_INTERVAL,
    DEFAULT_TOTAL_TIMESTEPS,
    DEFAULT_WARMUP_EPISODES,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
)
from txt2reward.core.log import configure_logging, get_logger
from txt2reward.training.callbacks import RewardEvolutionCallback
from txt2reward.training.device import detect_device
from txt2reward.training.env_factory import build_vec_env, make_env
from txt2reward.training.logger import TrainingLogger


def main() -> None:
    """CLI entry: PPO training with periodic LLM reward evolution.

    Parses arguments, optionally bootstraps ``reward_program.py``, builds
    vectorized envs, runs PPO with checkpoint + evolution callbacks, saves
    model/archive/log, and generates plots unless ``--no-plots``.

    Side effects:
        May delete artifacts when ``--fresh``; writes checkpoints, archive,
        log, and plot PNGs to disk.
    """
    import argparse

    configure_logging()
    log = get_logger("train")

    parser = argparse.ArgumentParser(description="Train PPO on highway-v0 with Text-to-Reward evolution")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument(
        "--reload-interval",
        type=int,
        default=DEFAULT_RELOAD_INTERVAL,
        help="Steps between reward_program.py reloads in each worker",
    )
    parser.add_argument(
        "--evolve-every",
        type=int,
        default=DEFAULT_EVOLVE_EVERY,
        help="Generate new reward every N episodes (after warmup)",
    )
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=DEFAULT_WARMUP_EPISODES,
        help="Episodes before first LLM reward generation",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=DEFAULT_DRIVING_GOAL,
        help="Natural language driving goal sent to the LLM",
    )
    parser.add_argument("--resume", type=str, default=None, metavar="PATH", help="Checkpoint .zip to resume from")
    parser.add_argument("--reward-path", type=str, default=REWARD_PROGRAM_PATH)
    parser.add_argument("--archive-file", type=str, default=ARCHIVE_FILE)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--log-file", type=str, default=LOG_FILE)
    parser.add_argument("--plot-dir", type=str, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--smooth", type=int, default=DEFAULT_PLOT_SMOOTH_WINDOW)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--bootstrap", action="store_true", help="Generate first reward program before training starts")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force a clean run: delete any existing log file, archive file, "
        "reward program and ppo_highway*.zip checkpoints before starting, "
        "instead of silently resuming.",
    )

    parser.add_argument(
        "--allow-dummy-env",
        action="store_true",
        help="Fall back to DummyVecEnv if SubprocVecEnv fails (single-process only).",
    )

    args = parser.parse_args()

    # Resolve paths so subprocess workers see the same files regardless of CWD.
    args.reward_path = os.path.abspath(args.reward_path)
    args.archive_file = os.path.abspath(args.archive_file)
    args.log_file = os.path.abspath(args.log_file)

    device = detect_device()
    log.info("[train] Using device: %s", device)
    log.info("[train] Driving goal: %s...", args.goal[:80])

    # ── Fresh start: wipe any local state so nothing gets resumed ────────────
    if args.fresh:
        targets = [args.log_file, args.archive_file, args.reward_path]
        for fname in targets:
            if os.path.exists(fname):
                os.remove(fname)
                log.info("[train] --fresh: removed %s", fname)
        for f in os.listdir("."):
            if f.startswith("ppo_highway") and f.endswith(".zip"):
                os.remove(f)
                log.info("[train] --fresh: removed checkpoint %s", f)

    # Logger before designer so episode count can resume evolution schedule.
    training_log = TrainingLogger(log_path=args.log_file)

    # ── Build the single RewardDesigner used for the whole run ────────────────
    from txt2reward.llm.designer import RewardDesigner, write_default_reward_program

    designer = RewardDesigner(
        goal=args.goal,
        evolve_every=args.evolve_every,
        warmup_episodes=args.warmup_episodes,
        reward_path=args.reward_path,
        archive_path=args.archive_file,
        initial_episode_count=training_log.episode_count(),
        initial_last_evolution_index=training_log.completed_evolution_index(args.warmup_episodes, args.evolve_every),
        verbose=True,
    )

    if training_log.episode_count() > 0:
        evo_idx = training_log.completed_evolution_index(args.warmup_episodes, args.evolve_every)
        log.info(
            "[train] Resuming evolution schedule from episode %s (last evolution index=%s).",
            training_log.episode_count(),
            evo_idx,
        )

    # ── Bootstrap: generate initial reward program if needed ──────────────────
    # Runs AFTER --fresh (if passed), so it only fires when no local
    # reward_program.py exists, or the user explicitly asked for a new one
    # via --bootstrap.
    if args.bootstrap or not os.path.exists(args.reward_path):
        log.info("[train] Bootstrapping initial reward program...")
        ok = designer.generate_reward()
        if ok:
            log.info("[train] Initial reward program generated successfully.")
        else:
            if not os.path.exists(args.reward_path):
                write_default_reward_program(args.reward_path)
                designer._current_code = designer._load_current_code()
                log.warning("[train] Bootstrap failed — wrote shipped default reward_program.py")
            else:
                log.warning("[train] Bootstrap failed — keeping existing reward_program.py")

    # ── Build environments ────────────────────────────────────────────────────
    env_fns = [
        make_env(rank=i, reload_interval=args.reload_interval, reward_path=args.reward_path) for i in range(args.n_envs)
    ]

    vec_env = build_vec_env(env_fns, allow_dummy_env=args.allow_dummy_env)

    # ── Build or restore PPO model ────────────────────────────────────────────
    if args.resume:
        model = PPO.load(args.resume, env=vec_env)
        log.info("[train] Resumed from checkpoint: %s", args.resume)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=device,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            tensorboard_log="./tb_logs/",
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=".",
        name_prefix="ppo_highway",
    )

    evolution_cb = RewardEvolutionCallback(
        designer=designer,
        logger=training_log,
        verbose=1,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info(
        "\n[train] Starting — %s timesteps | %s envs | evolve every %s episodes | warmup %s episodes",
        f"{args.timesteps:,}",
        args.n_envs,
        args.evolve_every,
        args.warmup_episodes,
    )

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=args.resume is None,
        callback=[checkpoint_cb, evolution_cb],
    )

    # ── Final save ────────────────────────────────────────────────────────────
    model.save("ppo_highway_txt2reward")
    training_log.save()

    log.info("\n[designer] Archive summary: %s", designer.archive.summary())

    vec_env.close()
    log.info(
        "\n[train] Done. Model + archive + log saved locally (ppo_highway_txt2reward.zip, %s, %s).",
        args.archive_file,
        args.log_file,
    )

    # ── Auto-generate plots ───────────────────────────────────────────────────
    if not args.no_plots:
        log.info("\n[train] Generating plots → '%s/' ...", args.plot_dir)
        try:
            from txt2reward.training.plots import generate_all_plots

            generate_all_plots(
                log_path=args.log_file,
                out_dir=args.plot_dir,
                smooth=args.smooth,
            )
            log.info("[train] Plots saved to %s/", args.plot_dir)
        except Exception as e:
            log.warning("[train] Plot generation failed: %s", e)
