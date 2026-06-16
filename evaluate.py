"""
evaluate.py
───────────
Evaluation of a PPO model trained on highway-v0 with Text-to-Reward.

Usage:
  python evaluate.py --model ppo_highway_txt2reward.zip
  python evaluate.py --model ppo_highway_txt2reward.zip --episodes 20 --render
  python evaluate.py --model ppo_highway_txt2reward.zip --generation 3

The evaluator runs the current reward_program.py by default.
Use --generation N to evaluate with a specific archived reward program.
Use --no-shaped to disable the shaped reward entirely.
"""

import os
import json
import argparse
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reward_wrapper import LLMRewardWrapper, REWARD_PROGRAM_PATH, _SPEED_SCALE
from reward_archive import RewardArchive, compute_fitness

ENV_CONFIG = {
    "vehicles_count":       30,
    "simulation_frequency": 15,
    "policy_frequency":      5,
    "duration":             60,
    "lanes_count":           4,
    "observation": {
        "type":           "Kinematics",
        "vehicles_count": 10,
        "features":       ["presence", "x", "y", "vx", "vy"],
        "normalize":      True,
        "absolute":       False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "reward_speed_range": [20, 30],
    "collision_reward":   -1.0,
    "high_speed_reward":   0.0,   # must match train.py so env_reward is comparable
    "right_lane_reward":   0.0,
    "lane_change_reward":  0.0,
}


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
            num_lanes   = ENV_CONFIG["lanes_count"],
            reward_path = reward_path,
        )
    env = Monitor(env)
    return env


def run_episode(model, env, deterministic: bool = True, render: bool = False) -> dict:
    obs, info = env.reset()
    if render:
        env.render()

    total_reward  = 0.0
    steps         = 0
    crashed       = False
    speed_sum     = 0.0
    overtakes     = 0
    lane_changes  = 0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps        += 1

        if render:
            env.render()

        if info.get("crashed", False):
            crashed = True

        # Speed normalization must match reward_wrapper._parse_full_obs
        # exactly (same threshold/scale) -- importing the constant instead
        # of re-deriving it here avoids the two copies silently diverging.
        vx_raw   = float(obs[0][3])
        speed_ms = max(0.0, vx_raw * _SPEED_SCALE) if abs(vx_raw) <= 1.5 else max(0.0, vx_raw)
        speed_sum += speed_ms

        # collect from episode_stats if available
        ep_stats = info.get("episode_stats", {})
        if ep_stats:
            overtakes    = ep_stats.get("total_overtakes",    0)
            lane_changes = ep_stats.get("total_lane_changes", 0)

        if terminated or truncated:
            break

    return {
        "total_reward":  float(total_reward),
        "steps":         steps,
        "crashed":       crashed,
        "mean_speed":    round(speed_sum / max(steps, 1), 2),
        "overtakes":     overtakes,
        "lane_changes":  lane_changes,
    }


def evaluate(
    model_path:    str,
    n_episodes:    int  = 10,
    use_shaped:    bool = True,
    render:        bool = False,
    deterministic: bool = True,
    save_path:     str | None = None,
    reward_path:   str = REWARD_PROGRAM_PATH,
) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[evaluate] Model not found: '{model_path}'")

    print(f"[evaluate] Loading model: {model_path}")
    if use_shaped:
        print(f"[evaluate] Reward program: {reward_path}")
    else:
        print("[evaluate] Shaped reward OFF — env reward only")

    render_mode = "human" if render else None
    env   = make_eval_env(use_shaped, render_mode, reward_path)
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

    crashes    = [r["crashed"]       for r in results]
    rewards    = [r["total_reward"]  for r in results]
    steps_list = [r["steps"]         for r in results]
    speeds     = [r["mean_speed"]    for r in results]
    overtakes  = [r["overtakes"]     for r in results]

    # Compute fitness for archive
    metrics = {
        "mean_speed":      float(np.mean(speeds)),
        "crash_rate":      float(np.mean(crashes)),
        "mean_overtakes":  float(np.mean(overtakes)),
        "mean_steps":      float(np.mean(steps_list)),
        "completion_rate": 1.0 - float(np.mean(crashes)),
        "max_steps":       300,
    }
    fitness = compute_fitness(metrics)

    summary = {
        "model_path":    model_path,
        "reward_path":   reward_path,
        "n_episodes":    n_episodes,
        "use_shaped":    use_shaped,
        "deterministic": deterministic,
        "episodes":      results,
        "mean_reward":   float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "min_reward":    float(np.min(rewards)),
        "max_reward":    float(np.max(rewards)),
        "mean_steps":    float(np.mean(steps_list)),
        "mean_speed":    float(np.mean(speeds)),
        "crash_rate":    float(np.mean(crashes)),
        "mean_overtakes": float(np.mean(overtakes)),
        "fitness":       fitness,
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
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model on highway-v0 (Text-to-Reward)"
    )
    parser.add_argument("--model",       type=str,
                        default="ppo_highway_txt2reward.zip")
    parser.add_argument("--episodes",    type=int,   default=10)
    parser.add_argument("--no-shaped",   action="store_true")
    parser.add_argument("--render",      action="store_true")
    parser.add_argument("--stochastic",  action="store_true")
    parser.add_argument("--save",        type=str,   default=None)
    parser.add_argument("--reward-path", type=str,   default=REWARD_PROGRAM_PATH,
        help="Path to reward_program.py to evaluate with")
    parser.add_argument("--generation",  type=int,   default=None,
        help="Evaluate using a specific archived generation (extracts to /tmp/)")
    parser.add_argument("--archive",     type=str,   default="reward_archive.json")

    args = parser.parse_args()

    reward_path = args.reward_path

    # Extract specific generation if requested
    if args.generation is not None:
        archive = RewardArchive(args.archive)
        entry   = archive.get_by_generation(args.generation)
        if entry is None:
            print(f"[evaluate] Generation {args.generation} not found in archive.")
            exit(1)
        tmp_path = f"/tmp/reward_gen{args.generation}.py"
        with open(tmp_path, "w") as f:
            f.write(entry["reward_code"])
        reward_path = tmp_path
        print(f"[evaluate] Using generation {args.generation} from archive.")

    evaluate(
        model_path    = args.model,
        n_episodes    = args.episodes,
        use_shaped    = not args.no_shaped,
        render        = args.render,
        deterministic = not args.stochastic,
        save_path     = args.save,
        reward_path   = reward_path,
    )
