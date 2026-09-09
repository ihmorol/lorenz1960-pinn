import numpy as np
import pandas as pd
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


# --------------------------------------------------------------------------
# per-epoch collocation breakdown
# --------------------------------------------------------------------------
def test_snapshots_are_off_by_default():
    """The run of record must be unaffected: no snapshot files, no extra cost."""
    from fydp2.train import train

    assert Config().snapshot_every == 0
    _, history = train(Config(depth=1, width=8, epochs=5, n_collocation=16,
                              log_every=5, eval_every=5, print_every=0))
    assert history.snapshots is None


def test_snapshot_rows_reconstruct_the_loss(tmp_path):
    """Every column is recorded, and loss_contribution sums to the reported loss."""
    from fydp2.history import SNAPSHOT_COLUMNS, load_snapshots, point_history
    from fydp2.train import train

    cfg = Config(depth=1, width=8, epochs=21, n_collocation=32, snapshot_every=10,
                 log_every=10, eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    _, history = train(cfg)

    frame = load_snapshots(tmp_path / "breakdown")
    assert list(frame.columns) == list(SNAPSHOT_COLUMNS)
    epochs = sorted(frame["epoch"].unique())
    assert epochs == [0, 10, 20]
    assert len(frame) == len(epochs) * cfg.n_collocation

    # loss_contribution = |r_i|^2 / (3 Nc), so it sums to mean(r^2) over all components.
    for epoch, group in frame.groupby("epoch"):
        expected = history.logged_loss[history.log_epoch.index(epoch)]
        assert group["loss_contribution"].sum() == pytest.approx(expected, rel=1e-5)

    # r = du/dt - f(u), column by column. The network trains in float32, so the
    # recorded residual agrees with a float64 reconstruction only to float32
    # precision -- the columns are consistent, not independently exact.
    for a in "xyz":
        assert frame[f"r_{a}"].to_numpy() == pytest.approx(
            (frame[f"d{a}_dt"] - frame[f"f_{a}"]).to_numpy(), abs=1e-6)

    one = point_history(tmp_path / "breakdown", 7)
    assert list(one["epoch"]) == epochs
    assert one["t"].nunique() == 1  # collocation points are fixed across training


def test_point_summary_and_residual_grid(tmp_path):
    from fydp2.history import residual_grid
    from fydp2.train import train, save_results

    cfg = Config(depth=1, width=8, epochs=31, n_collocation=32, snapshot_every=10,
                 log_every=10, eval_every=15, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    save_results(model, history, cfg)

    summary = pd.read_csv(tmp_path / "point_summary.csv")
    assert len(summary) == cfg.n_collocation
    assert (summary["r_max"] >= summary["r_final"]).all()
    assert summary["rank_by_final_residual"].nunique() == cfg.n_collocation

    epochs, centres, values = residual_grid(tmp_path / "breakdown", n_bins=12)
    assert values.shape == (len(epochs), 12) and centres.size == 12

    for name in ("residual_evolution", "residual_profiles", "point_convergence"):
        assert (tmp_path / "figures" / f"{name}.png").exists()


def test_run_summary_columns(tmp_path):
    from fydp2.train import save_results, train

    cfg = Config(depth=2, width=8, epochs=20, n_collocation=32, log_every=10,
                 eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    save_results(model, history, cfg)

    summary = pd.read_csv(tmp_path / "run_summary.csv")
    assert len(summary) == 1
    for column in ("arch", "depth", "width", "n_params", "rmse_combined_l2", "r2_x",
                   "err_norm_p99", "final_x", "final_ref_x", "residual_mse_collocation",
                   "residual_mse_dense", "residual_generalisation_gap",
                   "invariant_1_max_drift", "wall_clock_s", "epochs_to_1e-04"):
        assert column in summary.columns, column
    assert summary["arch"].iloc[0] == "2x8"
    # The reference solution conserves the invariants to solver precision.
    assert summary["invariant_1_max_drift_reference"].iloc[0] < 1e-9


def test_sweep_writes_runs_and_comparison(tmp_path):
    from fydp2 import sweep as sweep_module

    base = Config(epochs=12, n_collocation=16, log_every=6, eval_every=6, print_every=0,
                  runs_dir=str(tmp_path))
    original, sweep_module.SNAPSHOT_EVERY = sweep_module.SNAPSHOT_EVERY, 6
    try:
        comparison = sweep_module.sweep(base, depths=(1,), widths=(4, 6))
    finally:
        sweep_module.SNAPSHOT_EVERY = original

    assert list(comparison["arch"]) == ["1x4", "1x6"]
    assert (tmp_path / "comparison.csv").exists()
    for arch in ("1x4", "1x6"):
        run = tmp_path / arch
        assert (run / "metrics.csv").exists()
        assert (run / "run_summary.csv").exists()
        assert (run / "point_summary.csv").exists()
        assert (run / "history" / "pinn.pt").exists()
        assert len(list((run / "breakdown").glob("epoch_*.csv"))) == 3
    # A single-activation, single-seed sweep has nothing to say about either.
    written = {p.stem for p in (tmp_path / "figures").glob("*.png")}
    assert "architecture_heatmap" in written and "architecture_scatter" in written
    assert "activation_comparison" not in written and "seed_robustness" not in written


def test_sweep_leaves_the_run_of_record_alone(tmp_path):
    """The sweep must never write into src/fydp2/results or src/fydp2/history."""
    from fydp2.sweep import sweep_config

    base = Config(runs_dir=str(tmp_path))
    cfg = sweep_config(base, 5, 70)
    assert cfg.results_path == tmp_path / "5x70"
    assert cfg.ckpt_path == tmp_path / "5x70" / "history"
    assert cfg.snapshot_every > 0
    assert Config().results_path not in (cfg.results_path, cfg.ckpt_path)
