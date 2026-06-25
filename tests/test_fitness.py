from reward_archive import (
    compute_fitness,
    compute_fitness_v6,
    compute_fitness_v7,
    is_passive_driving,
)


def test_collision_reduces_fitness():
    metrics = {
        "crash_rate": 1.0,
        "mean_speed": 0.0,
        "mean_overtakes": 0.0,
        "mean_long_jerk": 1.0,
        "mean_ttc": 1.0,
        "completion_rate": 0.0,
    }
    assert compute_fitness(metrics) < 0.1


def test_passive_safe_scores_lower_than_active_safe():
    """0% crash at 20 m/s with no overtakes must score well below 25 m/s + overtakes."""
    passive = {
        "mean_speed": 20.0,
        "crash_rate": 0.0,
        "mean_overtakes": 0.0,
        "mean_long_jerk": 0.6,
        "mean_ttc": 2.4,
        "p10_ttc": 1.4,
        "min_ttc": 0.3,
        "completion_rate": 1.0,
    }
    active = {
        "mean_speed": 25.0,
        "crash_rate": 0.05,
        "mean_overtakes": 3.0,
        "mean_long_jerk": 1.0,
        "mean_ttc": 5.0,
        "p10_ttc": 4.0,
        "min_ttc": 2.0,
        "completion_rate": 0.95,
    }
    assert is_passive_driving(passive)
    assert not is_passive_driving(active)
    assert compute_fitness(passive) < compute_fitness(active) * 0.6


def test_passive_gate_inactive_while_still_crashing():
    """v6 passive gate stays off while crash_rate is high (ablation helper)."""
    from reward_archive import _passive_driving_gate

    assert _passive_driving_gate(20.0, 0.0, 0.5) == 1.0
    assert _passive_driving_gate(20.0, 0.0, 0.0) < 1.0
    assert not is_passive_driving({"crash_rate": 0.5, "mean_speed": 20.0, "mean_overtakes": 0.0})


def test_v7_transition_no_longer_beats_active():
    """RCA gen-4 transition profile must not outscore ideal active under v7."""
    transition = {
        "mean_speed": 21.2,
        "crash_rate": 0.35,
        "mean_overtakes": 0.25,
        "mean_long_jerk": 7.0,
        "mean_ttc": 1.47,
        "p10_ttc": 0.49,
        "min_ttc": 0.0001,
        "total_lane_changes": 401,
        "n_episodes": 40,
        "total_overtakes": 10,
    }
    ideal = {
        "mean_speed": 27.0,
        "crash_rate": 0.08,
        "mean_overtakes": 2.5,
        "mean_long_jerk": 2.0,
        "mean_ttc": 3.5,
        "p10_ttc": 2.5,
        "min_ttc": 1.0,
        "total_lane_changes": 12,
        "n_episodes": 40,
        "total_overtakes": 100,
    }
    assert compute_fitness_v7(transition, generation=4) < compute_fitness_v7(ideal, generation=5)
    assert compute_fitness_v6(transition) > compute_fitness_v6(ideal) * 0.15


def test_v7_slow_to_survive_trend_penalised():
    prev = {
        "mean_speed": 23.0,
        "crash_rate": 0.20,
        "mean_overtakes": 0.4,
        "mean_long_jerk": 5.0,
        "mean_ttc": 2.0,
        "p10_ttc": 1.0,
        "min_ttc": 0.5,
    }
    current = {
        "mean_speed": 20.5,
        "crash_rate": 0.10,
        "mean_overtakes": 0.1,
        "mean_long_jerk": 1.0,
        "mean_ttc": 2.5,
        "p10_ttc": 1.5,
        "min_ttc": 0.8,
    }
    with_trend = compute_fitness_v7(current, generation=5, prev_metrics=prev)
    without = compute_fitness_v7(current, generation=5, prev_metrics=None)
    assert with_trend < without
