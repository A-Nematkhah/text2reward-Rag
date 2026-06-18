from reward_archive import compute_fitness


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
