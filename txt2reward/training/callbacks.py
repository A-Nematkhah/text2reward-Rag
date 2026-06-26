"""Stable-Baselines3 callbacks for reward evolution."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from txt2reward.training.logger import TrainingLogger


def read_sb3_scalar(model, key: str, default: float = 0.0) -> float:
    """Read a float from SB3's internal logger, returning ``default`` on failure."""
    try:
        return float(model.logger.name_to_value.get(key, default))
    except Exception:
        return default


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
        completed: list[dict] = []

        for info in infos:
            stats = info.get("episode_stats")
            if stats is None:
                continue
            completed.append(stats)

        for stats in completed:
            meta_before = self.designer.get_weights()
            policy_snap = self.designer.get_policy_snapshot()

            self.training_logger.log_episode(
                stats=stats,
                timestep=self.num_timesteps,
                weights=meta_before,
                policy_snap=policy_snap,
            )
            self.designer.accumulate_episode(stats)

            if self.designer.maybe_evolve():
                meta_after = self.designer.get_weights()
                policy_snap = self.designer.get_policy_snapshot()
                evo_metrics = self.designer.get_last_evolution_metrics() or {}
                stats_window = {
                    "generation": meta_after.get("generation"),
                    "mean_speed": evo_metrics.get("mean_speed"),
                    "crash_rate": evo_metrics.get("crash_rate"),
                    "mean_overtakes": evo_metrics.get("mean_overtakes"),
                    "curriculum_phase": evo_metrics.get("curriculum_phase"),
                    "fitness": evo_metrics.get("fitness"),
                }
                self.training_logger.log_llm_update(
                    episode=self.training_logger._episode_n,
                    timestep=self.num_timesteps,
                    weights_before=meta_before,
                    weights_after=meta_after,
                    stats_window=stats_window,
                    policy_snap=policy_snap,
                )

        if completed:
            self.training_logger.save_periodically(every_n=10)

        return True

    def _on_rollout_end(self) -> None:
        entropy = -read_sb3_scalar(self.model, "train/entropy_loss", 0.0)
        val_loss = read_sb3_scalar(self.model, "train/value_loss", 0.0)
        pol_loss = read_sb3_scalar(self.model, "train/policy_gradient_loss", 0.0)
        expl_var = read_sb3_scalar(self.model, "train/explained_variance", 0.0)

        if entropy != 0.0 or val_loss != 0.0:
            self.designer.push_policy_metrics(
                entropy=entropy,
                value_loss=val_loss,
                policy_loss=pol_loss,
                explained_variance=expl_var,
            )
