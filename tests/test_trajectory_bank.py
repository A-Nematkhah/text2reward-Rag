"""Trajectory bank metric aggregation."""

from txt2reward.trajectory.bank import _aggregate_trajectory_metrics


def test_trajectory_metrics_include_robust_ttc():
    states = [
        {"speed_ms": 20.0, "collided": False, "overtook": False, "long_jerk": 0.0, "ttc": 30.0},
        {"speed_ms": 20.0, "collided": False, "overtook": False, "long_jerk": 0.0, "ttc": 1.0},
    ]
    m = _aggregate_trajectory_metrics(states)
    assert m["min_ttc"] == 1.0
    assert "p10_ttc" in m
