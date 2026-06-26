"""
evaluate.py
───────────
Evaluation of a PPO model trained on highway-v0 with Text-to-Reward.

Usage:
  python evaluate.py --model ppo_highway_txt2reward.zip
  python evaluate.py --model ppo_highway_txt2reward.zip --episodes 20 --render
  python evaluate.py --model ppo_highway_txt2reward.zip --generation 3
  python evaluate.py --model ppo_highway_50000_steps.zip --episodes 20 --render


The evaluator runs the current reward_program.py by default.
Use --generation N to evaluate with a specific archived reward program.
Use --no-shaped to disable the shaped reward entirely.
"""

import argparse
import json
import os

import numpy as np
from stable_baselines3 import PPO

from txt2reward.archive.archive import RewardArchive, compute_fitness
from txt2reward.core.log import configure_logging, get_logger
from txt2reward.core.metrics import aggregate_eval_fitness_metrics, denormalize_speed
from txt2reward.core.types import EvalEpisodeResult
from txt2reward.llm.validation import write_validated_reward_tempfile
from txt2reward.reward.wrapper import REWARD_PROGRAM_PATH
from txt2reward.training.env_factory import make_highway_env

log = get_logger("evaluate")


def run_episode(model, env, deterministic: bool = True, render: bool = False) -> EvalEpisodeResult:
    """Roll out one evaluation episode and collect behaviour metrics.

    Args:
        model: Loaded SB3 ``PPO`` policy.
        env: Highway env (typically from ``make_highway_env``).
        deterministic: Use mean action when True.
        render: Call ``env.render()`` each step when True.

    Returns:
        Episode totals and wrapper ``episode_stats`` fields for fitness.
    """
    obs, info = env.reset()
    if render:
        env.render()

    total_reward = 0.0
    steps = 0
    crashed = False
    speed_sum = 0.0
    ep_stats: dict = {}

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if render:
            env.render()

        if info.get("crashed", False):
            crashed = True

        vx_raw = float(obs[0][3])
        speed_ms = denormalize_speed(vx_raw)
        speed_sum += speed_ms

        if terminated or truncated:
            ep_stats = info.get("episode_stats") or {}
            break

    return {
        "total_reward": float(total_reward),
        "steps": steps,
        "crashed": crashed,
        "mean_speed": round(speed_sum / max(steps, 1), 2),
        "overtakes": ep_stats.get("total_overtakes", 0),
        "lane_changes": ep_stats.get("total_lane_changes", 0),
        "mean_ttc": ep_stats.get("mean_ttc", 30.0),
        "p10_ttc": ep_stats.get("p10_ttc", 30.0),
        "min_ttc": ep_stats.get("min_ttc", 30.0),
        "ttc_vals": list(ep_stats.get("ttc_vals", [])),
        "mean_long_jerk": ep_stats.get("mean_long_jerk", 0.0),
        "mean_accel": ep_stats.get("mean_accel", 0.0),
    }


def evaluate(
    model_path: str,
    n_episodes: int = 10,
    use_shaped: bool = True,
    render: bool = False,
    deterministic: bool = True,
    save_path: str | None = None,
    reward_path: str = REWARD_PROGRAM_PATH,
) -> dict:
    """Evaluate a checkpoint over multiple episodes.

    Args:
        model_path: Path to a ``.zip`` SB3 checkpoint.
        n_episodes: Number of rollouts.
        use_shaped: Wrap env with ``LLMRewardWrapper`` when True.
        render: Human render mode when True.
        deterministic: Greedy policy actions when True.
        save_path: Optional JSON path for aggregated results.
        reward_path: ``reward_program.py`` path for shaped reward.

    Returns:
        Dict with per-episode results and aggregate fitness metrics.

    Side effects:
        Loads model and env; logs progress; may write ``save_path``.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[evaluate] Model not found: '{model_path}'")

    log.info(f"[evaluate] Loading model: {model_path}")
    if use_shaped:
        log.info(f"[evaluate] Reward program: {reward_path}")
    else:
        log.info("[evaluate] Shaped reward OFF — env reward only")

    render_mode = "human" if render else None
    env = make_highway_env(
        render_mode=render_mode,
        use_shaped=use_shaped,
        reward_path=reward_path,
        monitor=True,
    )
    model = PPO.load(model_path, device="cpu")

    log.info(
        f"\n[evaluate] Running {n_episodes} episodes | "
        f"shaped={'ON' if use_shaped else 'OFF'} | "
        f"deterministic={deterministic}\n"
    )

    results = []
    for ep in range(1, n_episodes + 1):
        ep_result = run_episode(model, env, deterministic=deterministic, render=render)
        results.append(ep_result)
        log.info(
            f"  Episode {ep:3d}/{n_episodes} | "
            f"reward={ep_result['total_reward']:+7.3f} | "
            f"steps={ep_result['steps']:3d} | "
            f"speed={ep_result['mean_speed']:5.2f} m/s | "
            f"overtakes={ep_result['overtakes']:2d} | "
            f"crashed={'YES' if ep_result['crashed'] else ' no'}"
        )

    env.close()

    crashes = [r["crashed"] for r in results]
    rewards = [r["total_reward"] for r in results]
    steps_list = [r["steps"] for r in results]
    speeds = [r["mean_speed"] for r in results]
    overtakes = [r["overtakes"] for r in results]
    metrics = aggregate_eval_fitness_metrics(results)
    fitness = compute_fitness(metrics)

    summary = {
        "model_path": model_path,
        "reward_path": reward_path,
        "n_episodes": n_episodes,
        "use_shaped": use_shaped,
        "deterministic": deterministic,
        "episodes": results,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "mean_speed": float(np.mean(speeds)),
        "crash_rate": float(np.mean(crashes)),
        "mean_overtakes": float(np.mean(overtakes)),
        "fitness": fitness,
    }

    log.info("\n" + "─" * 65)
    log.info(f"  Mean reward   : {summary['mean_reward']:+.3f}  ± {summary['std_reward']:.3f}")
    log.info(f"  Min / Max     : {summary['min_reward']:+.3f}  /  {summary['max_reward']:+.3f}")
    log.info(f"  Mean steps    : {summary['mean_steps']:.1f}")
    log.info(f"  Mean speed    : {summary['mean_speed']:.2f} m/s")
    log.info(f"  Mean overtakes: {summary['mean_overtakes']:.2f} per episode")
    log.info(f"  Crash rate    : {float(np.mean(crashes)) * 100:.1f}%  ({sum(crashes)}/{n_episodes})")
    log.info(f"  Fitness score : {fitness:.4f}")
    log.info("─" * 65)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log.info(f"\n[evaluate] Results saved to '{save_path}'")

    return summary


def main() -> None:
    """CLI entry: load checkpoint, run evaluation rollouts, print fitness summary."""
    configure_logging()

    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on highway-v0 (Text-to-Reward)")
    parser.add_argument("--model", type=str, default="ppo_highway_txt2reward.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-shaped", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument(
        "--reward-path", type=str, default=REWARD_PROGRAM_PATH, help="Path to reward_program.py to evaluate with"
    )
    parser.add_argument(
        "--generation", type=int, default=None, help="Evaluate using a specific archived generation (extracts to /tmp/)"
    )
    parser.add_argument("--archive", type=str, default="reward_archive.json")

    args = parser.parse_args()

    reward_path = args.reward_path
    temp_reward_path: str | None = None

    # Extract specific generation if requested
    if args.generation is not None:
        archive = RewardArchive(os.path.abspath(args.archive))
        entry = archive.get_by_generation(args.generation)
        if entry is None:
            log.error(f"[evaluate] Generation {args.generation} not found in archive.")
            exit(1)
        try:
            temp_reward_path = write_validated_reward_tempfile(entry["reward_code"], args.generation)
        except ValueError as exc:
            log.error(f"[evaluate] {exc}")
            exit(1)
        reward_path = temp_reward_path
        log.info(f"[evaluate] Using generation {args.generation} from archive (validated).")

    try:
        evaluate(
            model_path=args.model,
            n_episodes=args.episodes,
            use_shaped=not args.no_shaped,
            render=args.render,
            deterministic=not args.stochastic,
            save_path=args.save,
            reward_path=reward_path,
        )
    finally:
        if temp_reward_path and os.path.exists(temp_reward_path):
            os.remove(temp_reward_path)
