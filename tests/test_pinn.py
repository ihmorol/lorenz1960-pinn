import numpy as np
import pytest
import torch

from fydp2.config import Config, reference_trajectory
from fydp2.pinn import PINN, ode_residual


def test_hard_ic_exact():
    model = PINN(Config())
    out = model(torch.zeros(1, 1))
    expected = torch.tensor([Config().initial_state])
    assert torch.allclose(out, expected, atol=1e-6)


def test_residual_zero_on_truth():
    t, ys = reference_trajectory(Config(), n=1001)
    dudt = np.gradient(ys, t, axis=0)
    r = ode_residual(
        torch.tensor(ys, dtype=torch.float64),
        torch.tensor(dudt, dtype=torch.float64),
        Config().coefficients,
    )
    assert r[1:-1].abs().max().item() < 1e-2


def test_training_reduces_loss():
    from fydp2.train import train

    cfg = Config(epochs=100, n_collocation=101, lbfgs_iters=50, seed=0)
    _, history = train(cfg)
    assert history.loss[-1] < 0.1 * history.loss[0]


def test_soft_ic_trains():
    from fydp2.train import train

    cfg = Config(ic="soft", epochs=200, n_collocation=101, lbfgs_iters=50, seed=0)
    _, history = train(cfg)
    assert history.loss[-1] < 0.1 * history.loss[0]


def test_history_records_diagnostics():
    from fydp2.train import train

    cfg = Config(epochs=50, n_collocation=64, lbfgs_iters=0, log_every=10, eval_every=25, seed=0)
    _, history = train(cfg)
    n = len(history.log_epoch)
    assert n == len(history.grad_norm) == len(history.residual_loss) == len(history.lr)
    assert all(len(v) == n for v in history.layer_grad_norms.values())
    assert history.adam_iters == len(history.loss) == cfg.epochs
    assert len(history.ref_epoch) == len(history.ref_mse) > 0
    assert all(g > 0 for g in history.grad_norm)


def test_figure_suite_writes_all_panels(tmp_path):
    from fydp2 import figures
    from fydp2.train import collect_artifacts, train

    cfg = Config(depth=1, width=8, epochs=30, n_collocation=64, lbfgs_iters=0,
                 log_every=10, eval_every=15, seed=0)
    model, history = train(cfg)
    written = figures.generate_all(collect_artifacts(model, history, cfg), tmp_path, ("png",))
    assert set(written) == {
        "training_dynamics", "gradient_diagnostics", "collocation_points",
        "solution_vs_reference", "error_analysis", "phase_portraits",
        "metrics_summary", "invariant_drift", "physics_residual",
    }
    assert all(p.exists() and p.stat().st_size > 0 for paths in written.values() for p in paths)


def test_run_report_writes_tables_and_figures(tmp_path):
    from fydp2 import figures
    from fydp2.train import collect_artifacts, train

    cfg = Config(depth=1, width=8, epochs=20, n_collocation=64, lbfgs_iters=0,
                 log_every=10, eval_every=10, seed=0)
    model, history = train(cfg)
    metrics = figures.write_run_report(collect_artifacts(model, history, cfg), tmp_path, ("png",))
    assert list(metrics["state"]) == ["x", "y", "z", "combined_l2"]
    for name in ("metrics.csv", "results.png", "loss_history.csv"):
        assert (tmp_path / name).exists()
    assert (tmp_path / "figures" / "training_dynamics.png").exists()


def test_history_round_trips_through_csv(tmp_path):
    from fydp2 import figures
    from fydp2.history import TrainHistory
    from fydp2.train import collect_artifacts, train

    cfg = Config(depth=1, width=8, epochs=20, n_collocation=64, lbfgs_iters=0,
                 log_every=5, eval_every=10, seed=0)
    model, history = train(cfg)
    data = tmp_path / "data"
    figures.write_run_report(collect_artifacts(model, history, cfg), tmp_path, ("png",), data)

    assert not (tmp_path / "loss_history.csv").exists()  # bulk telemetry stays out of results/
    reloaded = TrainHistory.from_saved(data)
    assert reloaded.loss == history.loss
    assert reloaded.log_epoch == history.log_epoch
    assert reloaded.grad_norm == pytest.approx(history.grad_norm)
    assert reloaded.layer_grad_norms.keys() == history.layer_grad_norms.keys()
    assert reloaded.ref_mse == pytest.approx(history.ref_mse)


def test_sweep_figures(tmp_path):
    import itertools

    import pandas as pd

    from fydp2 import figures

    rows = [
        {"depth": d, "width": w, "activation": a, "seed": s, "rmse": 10.0 ** -(d + s)}
        for d, w, a, s in itertools.product((1, 2), (20, 50), ("tanh", "relu"), (0, 1))
    ]
    written = figures.generate_sweep(pd.DataFrame(rows), tmp_path, "rmse", ("png",))
    assert set(written) == {"architecture_heatmap", "activation_comparison", "seed_robustness"}
    assert all(p.exists() for paths in written.values() for p in paths)
