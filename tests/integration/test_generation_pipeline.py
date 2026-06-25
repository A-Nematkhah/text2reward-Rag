from reward_designer import _full_validation_pipeline, DEFAULT_BOOTSTRAP_REWARD_BODY


def test_generation_pipeline_passes_full_validation():
    ok, err, _ = _full_validation_pipeline(DEFAULT_BOOTSTRAP_REWARD_BODY)
    assert ok, err
