"""
training_logger.py
──────────────────
Collects and stores training logs for plotting and analysis.

Updated for Text-to-Reward architecture.
The per_episode records now include 'generation' instead of weight components.
The llm_updates now record reward_code and archive metadata instead of weights.

Output file: training_log.json

Structure
─────────
{
  "meta": {
    "start_time", "elapsed_sec",
    "total_episodes", "total_llm_updates"
  },
  "per_episode": [
    {
      "episode", "timestep", "generation",
      "env_reward", "shaped_reward",
      "mean_speed", "mean_front_dist",
      "collisions", "steps", "crashed",
      "mean_ttc", "mean_rel_vel",
      "mean_long_jerk", "mean_lat_jerk",
      "mean_accel", "mean_density",
      "total_overtakes", "total_lane_changes",
      "policy_entropy", "policy_value_loss",
      "policy_loss", "policy_explained_variance"
    }
  ],
  "llm_updates": [
    {
      "episode", "timestep",
      "generation_before", "generation_after",
      "stats_window",
      "policy_snap"
    }
  ]
}
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

LOG_FILE = "training_log.json"


class TrainingLogger:

    def __init__(self, log_path: str = LOG_FILE):
        self.log_path    = log_path
        self._episode_n  = 0
        self.per_episode: list[dict] = []
        self.llm_updates: list[dict] = []
        self._start_time = time.time()

        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                self.per_episode = old.get("per_episode", [])
                self.llm_updates = old.get("llm_updates", [])
                if self.per_episode:
                    self._episode_n = self.per_episode[-1]["episode"]
                print(
                    f"[logger] Resumed from '{log_path}' "
                    f"({len(self.per_episode)} episodes, "
                    f"{len(self.llm_updates)} LLM updates)"
                )
            except Exception as e:
                print(f"[logger] Could not load old log: {e} — starting fresh")

    def log_episode(
        self,
        stats: dict,
        timestep: int,
        weights: dict[str, Any],   # now: {"generation": int, "reward_path": str}
        policy_snap: dict | None = None,
    ) -> None:
        self._episode_n += 1

        record: dict[str, Any] = {
            "episode":   self._episode_n,
            "timestep":  timestep,
            "generation": weights.get("generation", 0),

            "env_reward":      stats.get("total_env_reward",    0.0),
            "shaped_reward":   stats.get("total_shaped_reward", 0.0),
            "mean_speed":      stats.get("mean_speed",          0.0),
            "mean_front_dist": stats.get("mean_front_dist",     0.0),
            "collisions":      stats.get("collisions",          0),
            "steps":           stats.get("steps",               0),
            "crashed":         stats.get("collisions",          0) > 0,

            "mean_ttc":           stats.get("mean_ttc",           0.0),
            "mean_rel_vel":       stats.get("mean_rel_vel",       0.0),
            "mean_long_jerk":     stats.get("mean_long_jerk",     0.0),
            "mean_lat_jerk":      stats.get("mean_lat_jerk",      0.0),
            "mean_accel":         stats.get("mean_accel",         0.0),
            "mean_density":       stats.get("mean_density",       0.0),
            "total_overtakes":    stats.get("total_overtakes",    0),
            "total_lane_changes": stats.get("total_lane_changes", 0),

            "policy_entropy":            policy_snap["entropy"]            if policy_snap else None,
            "policy_value_loss":         policy_snap["value_loss"]         if policy_snap else None,
            "policy_loss":               policy_snap["policy_loss"]        if policy_snap else None,
            "policy_explained_variance": policy_snap["explained_variance"] if policy_snap else None,
        }

        self.per_episode.append(record)

    def log_llm_update(
        self,
        episode: int,
        timestep: int,
        weights_before: dict[str, Any],
        weights_after: dict[str, Any],
        stats_window: dict,
        policy_snap: dict | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "episode":           episode,
            "timestep":          timestep,
            "generation_before": weights_before.get("generation", 0),
            "generation_after":  weights_after.get("generation",  0),
            "stats_window":      dict(stats_window),
            "policy_snap":       dict(policy_snap) if policy_snap else None,
        }
        self.llm_updates.append(record)

    def save(self) -> None:
        data = {
            "meta": {
                "start_time":        self._start_time,
                "elapsed_sec":       round(time.time() - self._start_time, 1),
                "total_episodes":    len(self.per_episode),
                "total_llm_updates": len(self.llm_updates),
            },
            "per_episode": self.per_episode,
            "llm_updates": self.llm_updates,
        }
        tmp = self.log_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.log_path)

    def save_periodically(self, every_n: int = 10) -> None:
        if self._episode_n % every_n == 0:
            self.save()
