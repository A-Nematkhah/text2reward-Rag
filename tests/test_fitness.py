from reward_archive import compute_fitness, is_passive_driving


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
    """Don't penalise low speed while crash_rate is still high — agent is learning."""
    from reward_archive import _passive_driving_gate

    assert _passive_driving_gate(20.0, 0.0, 0.5) == 1.0
    assert _passive_driving_gate(20.0, 0.0, 0.0) < 1.0
    assert not is_passive_driving({"crash_rate": 0.5, "mean_speed": 20.0, "mean_overtakes": 0.0})
