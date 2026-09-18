import numpy as np
import pandas as pd
import pytest
import torch

from pinn.config import Config, reference_trajectory
from pinn.pinn import PINN, ode_residual


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
    from pinn.train import train

    cfg = Config(epochs=100, n_collocation=101, lbfgs_iters=50, seed=0)
    _, history = train(cfg)
    assert history.loss[-1] < 0.1 * history.loss[0]


def test_soft_ic_trains():
    from pinn.train import train

    cfg = Config(ic="soft", epochs=200, n_collocation=101, lbfgs_iters=50, seed=0)
    _, history = train(cfg)
    assert history.loss[-1] < 0.1 * history.loss[0]


def test_history_records_diagnostics():
    from pinn.train import train

    cfg = Config(epochs=50, n_collocation=64, lbfgs_iters=0, log_every=10, eval_every=25, seed=0)
    _, history = train(cfg)
    n = len(history.log_epoch)
    assert n == len(history.grad_norm) == len(history.residual_loss) == len(history.lr)
    assert all(len(v) == n for v in history.layer_grad_norms.values())
    assert history.adam_iters == len(history.loss) == cfg.epochs
    assert len(history.ref_epoch) == len(history.ref_mse) > 0
    assert all(g > 0 for g in history.grad_norm)


def test_figure_suite_writes_all_panels(tmp_path):
    from pinn import figures
    from pinn.train import collect_artifacts, train

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
    from pinn import figures
    from pinn.train import collect_artifacts, train

    cfg = Config(depth=1, width=8, epochs=20, n_collocation=64, lbfgs_iters=0,
                 log_every=10, eval_every=10, seed=0)
    model, history = train(cfg)
    metrics = figures.write_run_report(collect_artifacts(model, history, cfg), tmp_path, ("png",))
    assert list(metrics["state"]) == ["x", "y", "z", "combined_l2"]
    for name in ("metrics.csv", "results.png", "loss_history.csv"):
        assert (tmp_path / name).exists()
    assert (tmp_path / "figures" / "training_dynamics.png").exists()


def test_history_round_trips_through_csv(tmp_path):
    from pinn import figures
    from pinn.history import TrainHistory
    from pinn.train import collect_artifacts, train

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

    from pinn import figures

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
    from pinn.train import train

    assert Config().snapshot_every == 0
    _, history = train(Config(depth=1, width=8, epochs=5, n_collocation=16,
                              log_every=5, eval_every=5, print_every=0))
    assert history.snapshots is None


def test_snapshot_rows_reconstruct_the_loss(tmp_path):
    """Every column is recorded, and loss_contribution sums to the reported loss."""
    from pinn.history import SNAPSHOT_COLUMNS, load_snapshots, point_history
    from pinn.train import train

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
    from pinn.history import residual_grid
    from pinn.train import train, save_results

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
    from pinn.train import save_results, train

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
    from pinn import sweep as sweep_module

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
    """The sweep must never write into src/pinn/results or src/pinn/history."""
    from pinn.sweep import sweep_config

    base = Config(runs_dir=str(tmp_path))
    cfg = sweep_config(base, 5, 70)
    assert cfg.results_path == tmp_path / "5x70"
    assert cfg.ckpt_path == tmp_path / "5x70" / "history"
    assert cfg.snapshot_every > 0
    assert Config().results_path not in (cfg.results_path, cfg.ckpt_path)


# --------------------------------------------------------------------------
# entry-point helpers
# --------------------------------------------------------------------------
def test_config_for_round_trips_a_finished_run(tmp_path):
    """A run's own summary is enough to rebuild the Config its weights need."""
    from pinn.sweep import config_for
    from pinn.train import save_results, train

    cfg = Config(depth=3, width=12, activation="gelu", epochs=10, n_collocation=16,
                 log_every=5, eval_every=5, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    save_results(model, history, cfg)

    rebuilt = config_for(tmp_path)
    assert (rebuilt.depth, rebuilt.width, rebuilt.activation) == (3, 12, "gelu")
    assert rebuilt.ckpt_path == tmp_path / "history"


def test_load_run_reproduces_the_saved_prediction(tmp_path):
    from pinn.sweep import config_for
    from pinn.train import load_run, predict, save_results, train

    cfg = Config(depth=2, width=10, epochs=10, n_collocation=16, log_every=5,
                 eval_every=5, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    save_results(model, history, cfg)

    t = np.linspace(0.0, 1.0, 51)
    reloaded, _, reloaded_cfg = load_run(config_for(tmp_path))
    assert predict(reloaded, t) == pytest.approx(predict(model, t), abs=0)
    assert reloaded_cfg.depth == cfg.depth


def test_resume_skips_a_finished_run(tmp_path):
    """A completed run is reused, not retrained; an interrupted one is redone."""
    from pinn.sweep import run_one, sweep_config

    base = Config(epochs=8, n_collocation=16, log_every=4, eval_every=4,
                  print_every=0, runs_dir=str(tmp_path))
    cfg = sweep_config(base, 1, 6)
    first = run_one(cfg, resume=True)
    stamp = (cfg.results_path / "run_summary.csv").stat().st_mtime_ns

    second = run_one(cfg, resume=True)
    assert (cfg.results_path / "run_summary.csv").stat().st_mtime_ns == stamp
    assert second["rmse_combined_l2"].iloc[0] == first["rmse_combined_l2"].iloc[0]

    (cfg.results_path / "run_summary.csv").unlink()  # simulate an interrupted run
    assert run_one(cfg, resume=True)["arch"].iloc[0] == "1x6"


# --------------------------------------------------------------------------
# trajectory capture and the batching question
# --------------------------------------------------------------------------
def test_batched_gradient_equals_sum_of_per_point_gradients():
    """The central claim: one batched backward pass is not an approximation.

    The residual system is diagonal in the collocation points, so the batched
    gradient and the summed per-point gradients are the same vector up to float32
    summation round-off. If this test ever fails, the batching argument in the
    README is wrong and the loss has grown a cross-point coupling.
    """
    from pinn.trajectory import gradient_equivalence

    report = gradient_equivalence(Config(depth=1, width=8, n_collocation=32, seed=0))
    assert report["rel_loss_diff"] < 1e-5
    assert report["rel_grad_diff"] < 1e-4
    assert report["grad_cosine"] > 0.99999
    # Agreement must be at round-off scale, not merely "close".
    assert report["rel_grad_diff"] < report["roundoff_bound"]


def test_record_per_epoch_walks_the_training_grid(tmp_path):
    """Every epoch is recorded, on the same points the loss was computed on."""
    from pinn.trajectory import record_per_epoch
    from pinn.train import get_device, make_grid

    cfg = Config(depth=1, width=8, epochs=12, n_collocation=16, log_every=6,
                 eval_every=6, print_every=0, seed=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history, data = record_per_epoch(cfg, every=1)

    # The stock loop snapshots on epoch % snapshot_every == 0, plus the last one.
    assert list(data.epochs) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert data.pred.shape == (cfg.epochs, cfg.n_collocation, 3)
    assert data.t.shape == (cfg.n_collocation,)
    assert data.ref.shape == (cfg.n_collocation, 3)
    assert np.all(np.diff(data.t) >= 0)          # sorted, so rows join into a curve
    assert history.snapshots is not None         # recorded by the existing writer

    # And the recorded points are the training grid, in the same order.
    grid = make_grid(cfg, get_device()).cpu().numpy().reshape(-1)
    assert data.t == pytest.approx(np.sort(grid))


def test_record_per_epoch_honours_the_cadence(tmp_path):
    from pinn.trajectory import record_per_epoch

    cfg = Config(depth=1, width=8, epochs=20, n_collocation=16, log_every=10,
                 eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    _, _, data = record_per_epoch(cfg, every=5)
    assert list(data.epochs) == [0, 5, 10, 15, 19]


def test_trajectory_round_trips_through_csv_and_npz(tmp_path):
    """A breakdown directory and a saved npz describe the same trajectory."""
    from pinn.trajectory import load_from_breakdown, load_npz, save_npz
    from pinn.train import train

    cfg = Config(depth=1, width=8, epochs=11, n_collocation=16, snapshot_every=5,
                 log_every=10, eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    train(cfg)

    data = load_from_breakdown(tmp_path / "breakdown")
    assert list(data.epochs) == [0, 5, 10]
    assert data.n_frames == 3 and data.n_points == cfg.n_collocation

    # The npz must be lossless: it is the artefact the figures are redrawn from.
    path = save_npz(data, tmp_path / "t.npz")
    again = load_npz(path)
    assert again.epochs.tolist() == data.epochs.tolist()
    assert np.array_equal(again.pred, data.pred)
    assert np.array_equal(again.t, data.t)
    assert np.array_equal(again.ref, data.ref)

    # RMSE must fall over training, and the last frame must beat the first.
    rmse = data.rmse()
    assert rmse[-1] < rmse[0]


def test_trajectory_frame_reports_the_batch_extent(tmp_path):
    from pinn.trajectory import load_from_breakdown, trajectory_frame
    from pinn.train import train

    cfg = Config(depth=1, width=8, epochs=6, n_collocation=16, snapshot_every=5,
                 log_every=10, eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    train(cfg)
    frame = trajectory_frame(load_from_breakdown(tmp_path / "breakdown"))

    assert len(frame) == 2
    for name in "xyz":
        assert (frame[f"pred_{name}_max"] >= frame[f"pred_{name}_min"]).all()
        assert frame[f"spread_{name}"].to_numpy() == pytest.approx(
            (frame[f"pred_{name}_max"] - frame[f"pred_{name}_min"]).to_numpy())
    assert (frame["n_points"] == cfg.n_collocation).all()


def test_trajectory_figures_render(tmp_path):
    """Each figure draws the frames it was given, on axes that hold the data."""
    import matplotlib.pyplot as plt

    from pinn import trajectory_figures as tfig
    from pinn.trajectory import load_from_breakdown
    from pinn.train import train

    cfg = Config(depth=1, width=8, epochs=41, n_collocation=24, snapshot_every=10,
                 log_every=20, eval_every=20, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    train(cfg)
    data = load_from_breakdown(tmp_path / "breakdown")
    rmse = data.rmse()

    fig = tfig.fig_trajectory_3d_panels(data.epochs, data.pred, data.ref,
                                        n_panels=4, rmse=rmse, label=data.label)
    panels = fig.axes
    assert len(panels) == 4
    # Shared limits: every panel must show the same box, sized to hold the data.
    limits = {(round(a.get_xlim()[0], 9), round(a.get_xlim()[1], 9)) for a in panels}
    assert len(limits) == 1
    assert panels[0].get_xlim()[0] <= data.pred[..., 0].min()
    assert panels[0].get_xlim()[1] >= data.pred[..., 0].max()
    plt.close(fig)

    fig = tfig.fig_trajectory_projections(data.epochs, data.pred, data.ref,
                                          max_lines=3, thin=2)
    assert len(fig.axes) >= 3
    plt.close(fig)

    fig = tfig.fig_trajectory_time_slices(data.epochs, data.t, data.pred, data.ref,
                                          max_lines=3, thin=2)
    assert len(fig.axes) >= 3
    plt.close(fig)

    fig = tfig.fig_trajectory_extent(data.epochs, data.pred, data.ref, log_epoch=True)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_animation_writes_moving_frames(tmp_path):
    """The GIF must actually animate: frames are distinct, not one repeated image."""
    from PIL import Image, ImageSequence

    from pinn import trajectory_figures as tfig
    from pinn.trajectory import load_from_breakdown
    from pinn.train import train

    cfg = Config(depth=1, width=8, epochs=21, n_collocation=16, snapshot_every=5,
                 log_every=20, eval_every=20, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    train(cfg)
    data = load_from_breakdown(tmp_path / "breakdown")

    path = tfig.animate_trajectory(data.epochs, data.t, data.pred, data.ref,
                                   path=tmp_path / "anim.gif", n_frames=5, thin=2, dpi=60)
    frames = [np.asarray(f.convert("L"), dtype=float)
              for f in ImageSequence.Iterator(Image.open(path))]
    assert len(frames) == 5
    deltas = [np.abs(frames[i + 1] - frames[i]).mean() for i in range(len(frames) - 1)]
    assert min(deltas) > 0.0


def test_batch_size_comparison_matched_compute():
    """Every arm is scored on the full batch; only the step count differs."""
    from pinn.trajectory import batch_size_comparison

    cfg = Config(depth=1, width=8, n_collocation=32, seed=0)
    curves, timing = batch_size_comparison(cfg, batch_sizes=(0, 8), point_budget=320,
                                           max_wall_s=30.0)

    full = curves[curves["batch_size"] == "full"]
    mini = curves[curves["batch_size"] == "8"]
    assert set(curves.columns) == {"batch_size", "step", "points_evaluated",
                                   "loss", "batch_loss"}
    # The full-batch arm takes one step per epoch over all 32 points; the
    # mini-batch arm needs four times as many steps for the same point count.
    assert full["points_evaluated"].iloc[0] == cfg.n_collocation
    assert mini["points_evaluated"].iloc[0] == 8
    assert (curves["loss"] > 0).all()

    # Every row is scored on the whole collocation set, so for the full-batch arm
    # the score is that arm's own loss object. It is taken after the step, which
    # makes it exactly what the next step starts from.
    assert full["loss"].to_numpy()[:-1] == pytest.approx(
        full["batch_loss"].to_numpy()[1:], rel=1e-6)
    # A mini-batch arm's own loss is a different quantity entirely, which is the
    # reason the comparison is plotted on "loss" and not on what each step saw.
    assert not np.allclose(mini["loss"].to_numpy(), mini["batch_loss"].to_numpy(), rtol=1e-3)

    assert set(timing) == {"full", "8"}
    assert all(v > 0 for v in timing.values())



# --------------------------------------------------------------------------
# time marching
# --------------------------------------------------------------------------
def test_marching_stages_tile_the_window():
    """Stage grids are disjoint, inside their own stage, and total n_collocation."""
    from pinn.marching import MarchingConfig, segment_grid, segment_sizes

    cfg = MarchingConfig(n_segments=5, n_collocation=103, depth=1, width=8)
    sizes = segment_sizes(cfg)
    assert sizes.sum() == cfg.n_collocation
    assert sizes.max() - sizes.min() <= 1          # the remainder is spread, not dumped

    edges = cfg.edges()
    assert edges.size == cfg.n_segments + 1
    assert (np.diff(edges) > 0).all()

    seen = []
    for k in range(cfg.n_segments):
        t = segment_grid(cfg, k, "cpu").numpy().reshape(-1)
        assert len(t) == sizes[k]
        assert (t >= edges[k]).all() and (t <= edges[k + 1]).all()
        seen.append(t)
    union = np.concatenate(seen)
    assert len(np.unique(union)) == cfg.n_collocation   # no point trained twice


def test_marching_hard_ic_holds_at_every_stage():
    """u(t_k) = v_k exactly, for every stage, by construction."""
    import torch

    from pinn.marching import MarchingConfig, SegmentedPINN

    cfg = MarchingConfig(n_segments=4, depth=1, width=8)
    model = SegmentedPINN(cfg)
    for k in range(cfg.n_segments):
        anchor = torch.tensor([0.4 + 0.1 * k, 0.7 + 0.1 * k, 0.9 + 0.1 * k])
        model.enter_segment(k, anchor)
        got = model(model.edges[k].reshape(1, 1)).detach()
        assert torch.allclose(got[0], anchor, atol=1e-6)


def test_marching_anchor_is_the_previous_stage_endpoint():
    """The marching step: stage k's IC is stage k-1's own right endpoint."""
    import torch

    from pinn.marching import MarchingConfig, SegmentedPINN, _stage_anchor

    cfg = MarchingConfig(n_segments=3, depth=1, width=8, seed=0)
    model = SegmentedPINN(cfg)
    model.enter_segment(0, torch.tensor(cfg.initial_state))
    model.activate(0)
    endpoint = model(model.edges[1].reshape(1, 1)).detach()[0]
    anchor = _stage_anchor(model, 1)
    assert torch.allclose(endpoint, anchor, atol=1e-7)


def test_marching_prediction_is_piecewise():
    """Each point is evaluated with its own stage's transform, boundaries included."""
    import torch

    from pinn.marching import MarchingConfig, SegmentedPINN, predict_marching

    cfg = MarchingConfig(n_segments=4, depth=1, width=8)
    model = SegmentedPINN(cfg)
    for k in range(cfg.n_segments):
        model.enter_segment(k, torch.tensor(cfg.initial_state) + 0.01 * k)

    t = np.linspace(cfg.t_span[0], cfg.t_span[1], 41)
    pred = predict_marching(model, t)
    assert pred.shape == (41, 3)
    # A boundary belongs to the stage starting there, so u(0) is stage 0's anchor.
    assert pred[0] == pytest.approx(cfg.initial_state, abs=1e-6)
    assert np.isfinite(pred).all()


def test_marching_training_reduces_loss_and_reports_seams(tmp_path):
    """The loop runs, every stage improves, and both tables come back populated."""
    from pinn.marching import MarchingConfig, train_marching

    cfg = MarchingConfig(depth=1, width=8, n_segments=4, epochs=200, n_collocation=32,
                        log_every=50, eval_every=100, print_every=0, seed=0)
    model, history, frames = train_marching(cfg)

    assert len(history.loss) == cfg.total_epochs
    assert history.wall_clock_s > 0
    # Each stage has its own collocation set, so its loss is only comparable within
    # itself -- never across stages. 200 epochs, not fewer: Adam's opening steps on
    # a 55-parameter net overshoot the initial loss badly and it takes a couple of
    # hundred epochs for the stage to come back below where it started.
    per_stage = np.array(history.loss).reshape(cfg.n_segments, cfg.epochs)
    assert (per_stage[:, -1] < per_stage[:, 0]).all()

    stages = frames["stages"]
    assert list(stages["stage"]) == list(range(cfg.n_segments))
    assert (stages["stage_rmse"] > 0).all()
    # Stage k+1 starts where stage k ended.
    assert stages["t_end"].iloc[:-1].to_numpy() == pytest.approx(
        stages["t_start"].iloc[1:].to_numpy())

    jumps = frames["boundary_jumps"]
    assert len(jumps) == cfg.n_segments - 1
    assert (jumps["jump_max"] >= 0).all()


def test_marching_summary_is_comparable_to_the_single_domain_row():
    """The summary carries the same column names the run summary uses."""
    from pinn.marching import MarchingConfig, marching_summary, train_marching

    cfg = MarchingConfig(depth=1, width=8, n_segments=3, epochs=40, n_collocation=24,
                         log_every=20, eval_every=20, print_every=0, seed=0)
    model, history, frames = train_marching(cfg)
    row = marching_summary(model, history, frames, cfg)

    assert len(row) == 1
    assert row["n_segments"].iloc[0] == 3
    assert row["epochs"].iloc[0] == cfg.total_epochs
    for column in ("rmse_combined_l2", "rmse_x", "mae_y", "max_abs_error_z",
                   "rel_mae_x", "final_loss", "max_boundary_jump", "wall_clock_s"):
        assert column in row.columns, column
    assert np.isfinite(row["rmse_combined_l2"].iloc[0])


def test_marching_finetune_moves_the_anchors():
    """With finetune_epochs > 0 the anchors are optimised, so the seams can close."""
    import torch

    from pinn.marching import MarchingConfig, train_marching

    cfg = MarchingConfig(depth=1, width=8, n_segments=3, epochs=40, n_collocation=24,
                         finetune_epochs=30, log_every=20, eval_every=20,
                         print_every=0, seed=0)
    model, history, _ = train_marching(cfg)

    assert len(history.loss) == cfg.total_epochs == 3 * 40 + 30
    assert model.anchors.requires_grad is False     # left frozen again afterwards
    # The fine-tune phase is the only thing that may move anchors 1 and 2, so
    # neither may still hold its initialised zeros.
    assert not torch.allclose(model.anchors[1], torch.zeros(3), atol=1e-12)


def test_marching_rejects_soft_ic_with_finetune():
    from pinn.marching import MarchingConfig

    with pytest.raises(ValueError, match="not supported"):
        MarchingConfig(ic="soft", finetune_epochs=10)
    MarchingConfig(ic="soft")           # sequential soft IC on its own is fine
    with pytest.raises(ValueError):
        MarchingConfig(n_segments=0)


def test_marching_finetune_never_moves_the_initial_condition():
    """The fine-tune closes interior seams; it must not re-solve a different IVP."""
    import torch

    from pinn.marching import MarchingConfig, predict_marching, train_marching

    cfg = MarchingConfig(depth=1, width=8, n_segments=3, epochs=100, n_collocation=24,
                         finetune_epochs=200, log_every=50, eval_every=100,
                         print_every=0, seed=0)
    model, _, _ = train_marching(cfg)

    assert torch.allclose(model.anchors[0].detach(),
                          torch.tensor(cfg.initial_state), atol=1e-6)
    # And the assembled solution still starts at the stated initial condition.
    assert predict_marching(model, np.array([cfg.t_span[0]]))[0] == pytest.approx(
        cfg.initial_state, abs=1e-6)
