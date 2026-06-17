from neural_latent_lab.training.metrics import RunMetrics


def test_finalize_ratios():
    m = RunMetrics(
        experiment_id="x", config_name="c", model_type="latent", dataset="addition",
        seed=0, latent_depth_k=4, num_parameters=1000,
        test_accuracy=0.8, estimated_flops=1e6, tokens_generated=4.0,
    )
    m.finalize_ratios()
    assert abs(m.accuracy_per_flop - 0.8 / 1e6) < 1e-18
    assert abs(m.accuracy_per_token - 0.2) < 1e-9


def test_to_dict_contains_required_fields():
    m = RunMetrics(
        experiment_id="x", config_name="c", model_type="baseline", dataset="addition",
        seed=0, latent_depth_k=0, num_parameters=10,
    )
    d = m.to_dict()
    for field in ("experiment_id", "test_accuracy", "ood_accuracy", "estimated_flops"):
        assert field in d
