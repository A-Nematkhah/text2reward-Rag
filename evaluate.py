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

import os
import json
import argparse
import tempfile
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env_config import ENV_CONFIG
from reward_wrapper import LLMRewardWrapper, REWARD_PROGRAM_PATH
from reward_archive import RewardArchive, compute_fitness
from reward_sandbox import validate_reward_code

# highway-env normalises vx into [-1, 1] using the range [-2*MAX_SPEED, 2*MAX_SPEED]
# with Vehicle.MAX_SPEED = 40.0 m/s, so the de-normalisation factor is 2*40 = 80,
# matching the corrected _SPEED_SCALE in reward_wrapper.py. This used to be 40.0
# here too, which silently halved every reported mean_speed.
_SPEED_SCALE = 80.0


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 30.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _pool_ttc_metrics(episode_results: list[dict]) -> tuple[float, float]:
    """Pool step-level TTC values across episodes (matches training aggregation)."""
    all_ttc: list[float] = []
    for r in episode_results:
        if r.get("ttc_vals"):
            all_ttc.extend(float(v) for v in r["ttc_vals"])
        else:
            all_ttc.append(float(r.get("min_ttc", 30.0)))

    if all_ttc:
        return _percentile(all_ttc, 10), min(all_ttc)

    p10_vals = [float(r.get("p10_ttc", 30.0)) for r in episode_results]
    min_vals = [float(r.get("min_ttc", 30.0)) for r in episode_results]
    return (
        float(np.mean(p10_vals)) if p10_vals else 30.0,
        float(np.min(min_vals)) if min_vals else 30.0,
    )


def _write_validated_archive_reward(entry: dict, generation: int) -> str:
    """Validate archived reward code before writing to a temp file for evaluation."""
    from reward_designer import _full_validation_pipeline

    code = entry["reward_code"]
    ok, err = validate_reward_code(code)
    if not ok:
        raise ValueError(f"Generation {generation} failed AST validation: {err}")

    smoke_ok, smoke_err, _ = _full_validation_pipeline(code)
    if not smoke_ok:
        raise ValueError(f"Generation {generation} failed smoke test: {smoke_err}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=f"_reward_gen{generation}.py", prefix="txt2reward_")
    os.close(tmp_fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(code)
    return tmp_path


def make_eval_env(
    use_shaped: bool,
    render_mode: str | None,
    reward_path: str = REWARD_PROGRAM_PATH,
):
    config = dict(ENV_CONFIG)
    env = gym.make("highway-v0", render_mode=render_mode, config=config)
    if use_shaped:
        env = LLMRewardWrapper(
            env,
            num_lanes=ENV_CONFIG["lanes_count"],
            reward_path=reward_path,
        )
    env = Monitor(env)
    return env


def run_episode(model, env, deterministic: bool = True, render: bool = False) -> dict:
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
        speed_ms = max(0.0, vx_raw * _SPEED_SCALE) if abs(vx_raw) <= 1.5 else max(0.0, vx_raw)
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[evaluate] Model not found: '{model_path}'")

    print(f"[evaluate] Loading model: {model_path}")
    if use_shaped:
        print(f"[evaluate] Reward program: {reward_path}")
    else:
        print("[evaluate] Shaped reward OFF — env reward only")

    render_mode = "human" if render else None
    env = make_eval_env(use_shaped, render_mode, reward_path)
    model = PPO.load(model_path, device="cpu")

    print(
        f"\n[evaluate] Running {n_episodes} episodes | "
        f"shaped={'ON' if use_shaped else 'OFF'} | "
        f"deterministic={deterministic}\n"
    )

    results = []
    for ep in range(1, n_episodes + 1):
        ep_result = run_episode(model, env, deterministic=deterministic, render=render)
        results.append(ep_result)
        print(
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
    p10_ttc, min_ttc = _pool_ttc_metrics(results)

    # Compute fitness (same metric fields as training/archive)
    metrics = {
        "mean_speed": float(np.mean(speeds)),
        "crash_rate": float(np.mean(crashes)),
        "mean_overtakes": float(np.mean(overtakes)),
        "mean_steps": float(np.mean(steps_list)),
        "completion_rate": 1.0 - float(np.mean(crashes)),
        "mean_ttc": float(np.mean([r.get("mean_ttc", 30.0) for r in results])),
        "p10_ttc": float(p10_ttc),
        "min_ttc": float(min_ttc),
        "mean_long_jerk": float(np.mean([r.get("mean_long_jerk", 0.0) for r in results])),
        "mean_accel": float(np.mean([r.get("mean_accel", 0.0) for r in results])),
        "max_steps": 300,
    }
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

    print("\n" + "─" * 65)
    print(f"  Mean reward   : {summary['mean_reward']:+.3f}  ± {summary['std_reward']:.3f}")
    print(f"  Min / Max     : {summary['min_reward']:+.3f}  /  {summary['max_reward']:+.3f}")
    print(f"  Mean steps    : {summary['mean_steps']:.1f}")
    print(f"  Mean speed    : {summary['mean_speed']:.2f} m/s")
    print(f"  Mean overtakes: {summary['mean_overtakes']:.2f} per episode")
    print(f"  Crash rate    : {summary['crash_rate']*100:.1f}%  ({sum(crashes)}/{n_episodes})")
    print(f"  Fitness score : {fitness:.4f}")
    print("─" * 65)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[evaluate] Results saved to '{save_path}'")

    return summary


if __name__ == "__main__":
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

    # Extract specific generation if requested
    if args.generation is not None:
        archive = RewardArchive(os.path.abspath(args.archive))
        entry = archive.get_by_generation(args.generation)
        if entry is None:
            print(f"[evaluate] Generation {args.generation} not found in archive.")
            exit(1)
        try:
            reward_path = _write_validated_archive_reward(entry, args.generation)
        except ValueError as exc:
            print(f"[evaluate] {exc}")
            exit(1)
        print(f"[evaluate] Using generation {args.generation} from archive (validated).")

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        use_shaped=not args.no_shaped,
        render=args.render,
        deterministic=not args.stochastic,
        save_path=args.save,
        reward_path=reward_path,
    )
