"""
train.py
────────
Main entry point for PPO training on highway-v0 with Text-to-Reward evolution.

New architecture vs original weight-based system
────────────────────────────────────────────────
  OLD: Natural Language Goal → LLM → reward_weights.json → PPO
  NEW: Natural Language Goal → LLM → reward_program.py → PPO → Critique → Loop

Changes
───────
  - RewardDesigner now generates complete reward functions (not just weights).
  - reward_program.py is the shared state between main process and workers.
  - LLMRewardWrapper loads compute_reward() dynamically from reward_program.py.
  - RewardEvolutionCallback drives critique + generation every N episodes.
  - RewardArchive persists all generations with metrics and fitness scores.
  - PPO policy health metrics are still harvested and passed to the designer.

Arguments new in this version
──────────────────────────────
  --goal          Natural language driving goal sent to the LLM
  --archive-file  Path to reward archive JSON
  --reward-path   Path to the generated reward_program.py
"""

from __future__ import annotations

import os
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from reward_wrapper import LLMRewardWrapper, REWARD_PROGRAM_PATH
from training_logger import TrainingLogger

# ── Environment configuration ─────────────────────────────────────────────────
ENV_CONFIG = {
    "vehicles_count": 30,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 60,
    "lanes_count": 4,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "reward_speed_range": [20, 30],
    "collision_reward": -1.0,
    "high_speed_reward": 0.0,
    "right_lane_reward": 0.0,
    "lane_change_reward": 0.0,
}


def make_env(rank: int = 0, reload_interval: int = 200, reward_path: str = REWARD_PROGRAM_PATH):
    def _init():
        env = gym.make("highway-v0", config=ENV_CONFIG)
        env = LLMRewardWrapper(
            env,
            reload_interval=reload_interval,
            num_lanes=ENV_CONFIG["lanes_count"],
            reward_path=reward_path,
        )
        env = Monitor(env)
        return env

    return _init


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _read_sb3_scalar(model, key: str, default: float = 0.0) -> float:
    try:
        return float(model.logger.name_to_value.get(key, default))
    except Exception:
        return default


# ── RewardEvolutionCallback ───────────────────────────────────────────────────


class RewardEvolutionCallback(BaseCallback):
    """
    SB3 callback — main process only.

    _on_step():        collect episode_stats, drive designer evolution
    _on_rollout_end(): harvest PPO health metrics → designer buffer
    """

    def __init__(
        self,
        designer,
        logger: TrainingLogger,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.designer = designer
        self.training_logger = logger

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            stats = info.get("episode_stats")
            if stats is None:
                continue

            meta_before = self.designer.get_weights()
            updated = self.designer.record_episode(stats)
            meta_after = self.designer.get_weights()

            policy_snap = self.designer.get_policy_snapshot()

            self.training_logger.log_episode(
                stats=stats,
                timestep=self.num_timesteps,
                weights=meta_after,
                policy_snap=policy_snap,
            )

            if updated:
                self.training_logger.log_llm_update(
                    episode=self.training_logger._episode_n,
                    timestep=self.num_timesteps,
                    weights_before=meta_before,
                    weights_after=meta_after,
                    stats_window={"generation": meta_after.get("generation")},
                    policy_snap=policy_snap,
                )

            self.training_logger.save_periodically(every_n=10)

        return True

    def _on_rollout_end(self) -> None:
        """Harvest PPO policy health after each gradient-update round."""
        entropy = -_read_sb3_scalar(self.model, "train/entropy_loss", 0.0)
        val_loss = _read_sb3_scalar(self.model, "train/value_loss", 0.0)
        pol_loss = _read_sb3_scalar(self.model, "train/policy_gradient_loss", 0.0)
        expl_var = _read_sb3_scalar(self.model, "train/explained_variance", 0.0)

        if entropy != 0.0 or val_loss != 0.0:
            self.designer.push_policy_metrics(
                entropy=entropy,
                value_loss=val_loss,
                policy_loss=pol_loss,
                explained_variance=expl_var,
            )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO on highway-v0 with Text-to-Reward evolution")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--reload-interval", type=int, default=200, help="Steps between reward_program.py reloads in each worker"
    )
    parser.add_argument(
        "--evolve-every", type=int, default=20, help="Generate new reward every N episodes (after warmup)"
    )
    parser.add_argument("--warmup-episodes", type=int, default=40, help="Episodes before first LLM reward generation")
    parser.add_argument(
        "--goal",
        type=str,
        default="Drive fast and safely on a 4-lane highway. Overtake slow vehicles. "
        "Avoid collisions. Prefer speeds above 25 m/s. Minimise harsh braking.",
        help="Natural language driving goal sent to the LLM",
    )
    parser.add_argument("--resume", type=str, default=None, metavar="PATH", help="Checkpoint .zip to resume from")
    parser.add_argument("--reward-path", type=str, default=REWARD_PROGRAM_PATH)
    parser.add_argument("--archive-file", type=str, default="reward_archive.json")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--log-file", type=str, default="training_log.json")
    parser.add_argument("--plot-dir", type=str, default="plots")
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--bootstrap", action="store_true", help="Generate first reward program before training starts")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force a clean run: delete any existing log file, archive file, "
        "reward program and ppo_highway*.zip checkpoints before starting, "
        "instead of silently resuming.",
    )

    args = parser.parse_args()

    device = _detect_device()
    print(f"[train] Using device: {device}")
    print(f"[train] Driving goal: {args.goal[:80]}...")

    # ── Fresh start: wipe any local state so nothing gets resumed ────────────
    if args.fresh:
        targets = [args.log_file, args.archive_file, args.reward_path]
        for fname in targets:
            if os.path.exists(fname):
                os.remove(fname)
                print(f"[train] --fresh: removed {fname}")
        for f in os.listdir("."):
            if f.startswith("ppo_highway") and f.endswith(".zip"):
                os.remove(f)
                print(f"[train] --fresh: removed checkpoint {f}")

    # ── Build the single RewardDesigner used for the whole run ────────────────
    # (Previously a separate throwaway `bootstrap_designer` was constructed
    # for the bootstrap step and a second one for training. Two instances
    # reading/writing the same files is unnecessary and was a contributing
    # factor to generation-tracking confusion. One instance, one source of
    # truth: self.archive.entries.)
    from reward_designer import RewardDesigner

    designer = RewardDesigner(
        goal=args.goal,
        evolve_every=args.evolve_every,
        warmup_episodes=args.warmup_episodes,
        reward_path=args.reward_path,
        archive_path=args.archive_file,
        verbose=True,
    )

    # ── Bootstrap: generate initial reward program if needed ──────────────────
    # Runs AFTER --fresh (if passed), so it only fires when no local
    # reward_program.py exists, or the user explicitly asked for a new one
    # via --bootstrap.
    if args.bootstrap or not os.path.exists(args.reward_path):
        print("[train] Bootstrapping initial reward program...")
        ok = designer.generate_reward()
        if ok:
            print("[train] Initial reward program generated successfully.")
        else:
            print("[train] Bootstrap failed — using default reward_program.py")

    # ── Build environments ────────────────────────────────────────────────────
    env_fns = [
        make_env(rank=i, reload_interval=args.reload_interval, reward_path=args.reward_path) for i in range(args.n_envs)
    ]

    try:
        vec_env = SubprocVecEnv(env_fns)
        print(f"[train] Using SubprocVecEnv with {args.n_envs} workers")
    except Exception as e:
        print(f"[train] SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        vec_env = DummyVecEnv(env_fns)

    # ── Build or restore PPO model ────────────────────────────────────────────
    if args.resume:
        model = PPO.load(args.resume, env=vec_env)
        print(f"[train] Resumed from checkpoint: {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=device,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            tensorboard_log="./tb_logs/",
        )

    training_log = TrainingLogger(log_path=args.log_file)

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
    print(
        f"\n[train] Starting — {args.timesteps:,} timesteps | "
        f"{args.n_envs} envs | evolve every {args.evolve_every} episodes | "
        f"warmup {args.warmup_episodes} episodes"
    )

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=args.resume is None,
        callback=[checkpoint_cb, evolution_cb],
    )

    # ── Final save ────────────────────────────────────────────────────────────
    model.save("ppo_highway_txt2reward")
    training_log.save()

    print(f"\n[designer] Archive summary: {designer.archive.summary()}")

    vec_env.close()
    print(
        f"\n[train] Done. Model + archive + log saved locally "
        f"(ppo_highway_txt2reward.zip, {args.archive_file}, {args.log_file})."
    )

    # ── Auto-generate plots ───────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\n[train] Generating plots → '{args.plot_dir}/' ...")
        try:
            from plot_training import generate_all_plots

            generate_all_plots(
                log_path=args.log_file,
                out_dir=args.plot_dir,
                smooth=args.smooth,
            )
            print(f"[train] Plots saved to {args.plot_dir}/")
        except Exception as e:
            print(f"[train] Plot generation failed: {e}")
