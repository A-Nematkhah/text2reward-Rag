from fitness import compute_fitness

def test_collision_reduces_fitness():
    metrics={'collision_rate':1.0,'efficiency_score':1.0,'comfort_score':1.0}
    assert compute_fitness(metrics) < 0.1
