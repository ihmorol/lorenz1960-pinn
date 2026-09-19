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
# sequential walk over the collocation points
# --------------------------------------------------------------------------
def test_sequential_rollout_matches_the_literal_point_by_point_loop():
    """The closed form is the loop: same quantity, float32 round-off apart.

    This is the load-bearing equivalence. The scheme is *defined* as walking the
    points in time order and integrating each interval with the trapezoid rule on
    the network's two endpoint outputs; the implementation computes it as a
    cumulative sum. If these ever disagree beyond round-off, the implementation is
    no longer the scheme.
    """
    import torch

    from pinn.pinn import PINN, sequential_rollout

    cfg = Config(sequential=True, depth=1, width=8)
    torch.manual_seed(0)
    model = PINN(cfg)
    t = torch.tensor([[0.30], [0.05], [0.90], [0.45], [0.12], [0.70], [0.60]])
    n = model.net(t).detach()

    order = torch.argsort(t.reshape(-1))
    times, raw = t.reshape(-1)[order], n[order]
    walked, prev_t, prev_n = [model.u0.reshape(3).clone()], model.t0, None
    for i in range(times.numel()):
        slope = raw[i] if i == 0 else 0.5 * (prev_n + raw[i])
        walked.append(walked[-1] + (times[i] - prev_t) * slope)
        prev_t, prev_n = times[i], raw[i]

    got = sequential_rollout(model, t, n)[order]
    assert torch.allclose(got, torch.stack(walked[1:]), atol=1e-7, rtol=0)
    # and it is genuinely a walk, not an independent per-point evaluation
    assert not torch.allclose(got, model.u0.reshape(3).expand_as(got), atol=1e-3)


def test_sequential_initial_condition_is_exact_and_unsorted_input_is_handled():
    """u(t0) is the stated IC to the bit, whatever order the points arrive in."""
    import torch

    from pinn.pinn import PINN

    cfg = Config(sequential=True, depth=1, width=8)
    model = PINN(cfg)
    assert model.sequential is True

    at_t0 = model(torch.tensor([[model.t0]]))
    assert torch.equal(at_t0.reshape(3), model.u0.reshape(3))

    # Shuffling the collocation points must not change the walked trajectory,
    # because the walk is defined on time order, not on storage order.
    t = torch.rand(50, 1)
    u = model(t)
    perm = torch.randperm(50)
    assert torch.allclose(model(t[perm]), u[perm], atol=1e-6)


def test_sequential_derivative_is_the_walk_step():
    """(u_i - u_{i-1})/dt == s_i identically, so dudt needs no autograd.

    Here s_i is the interval's quadrature slope: the trapezoid average of the two
    endpoint outputs, or n_1 itself on the first interval. Checked in float64. The
    difference quotient cancels two nearly-equal states and then divides by a
    small dt, so in float32 the round-off is amplified by roughly eps/dt and
    swamps the identity even though the identity itself is exact.
    """
    import torch

    from pinn.pinn import PINN, residual_parts

    cfg = Config(sequential=True, depth=1, width=8)
    model = PINN(cfg).double()
    t = torch.sort(torch.rand(40, 1, dtype=torch.float64), dim=0).values
    parts = residual_parts(model, t.clone().requires_grad_(True))

    slope = torch.cat([parts.n[:1], 0.5 * (parts.n[:-1] + parts.n[1:])])
    assert torch.equal(parts.dudt, slope)
    assert torch.allclose(parts.r, slope - parts.f, atol=1e-7)
    prev = torch.cat([model.u0, parts.u[:-1]])
    dt = t - torch.cat([torch.tensor([[model.t0]], dtype=torch.float64), t[:-1]])
    assert torch.allclose((parts.u - prev) / dt, slope, atol=1e-10)


def test_sequential_walk_has_a_second_order_error_floor():
    """The walk is implicit trapezoid: even fed exact slopes it leaves an O(dt^2) error.

    Feeding the true right-hand side in as if the network had produced it isolates
    the walk's own truncation error from any training error, and shows it is exactly
    second order: the error drops fourfold when dt halves, at every refinement.

    This matters because the walk re-integrates on whatever grid it is evaluated
    on, so the read-out grid still sets a floor under the scheme's accuracy — but a
    second-order one. At the 1001-point grid this project reports on, that floor is
    ~2e-07 (measured 1.7e-07), orders of magnitude below the single-domain run's
    training error, so the read-out grid no longer caps the accuracy a walked run
    can report.
    """
    import torch

    from pinn.config import reference_at
    from pinn.pinn import PINN, ode_rhs, sequential_rollout

    cfg = Config(sequential=True, depth=1, width=8)
    model = PINN(cfg).double()
    errors = []
    for n in (250, 500, 1000, 2000):
        t = torch.tensor(np.linspace(*cfg.t_span, n + 1)[1:].reshape(-1, 1))
        truth = torch.tensor(reference_at(cfg, t.reshape(-1).numpy()), dtype=torch.float64)
        walked = sequential_rollout(model, t, ode_rhs(truth, model.coeffs))
        errors.append(float((walked - truth).abs().max()))

    assert errors[0] > errors[1] > errors[2] > errors[3]   # a finer grid helps
    assert 3.9 < errors[0] / errors[1] < 4.1               # and it helps at exactly
    assert 3.9 < errors[1] / errors[2] < 4.1               # second order
    assert 3.9 < errors[2] / errors[3] < 4.1


def test_sequential_training_reduces_loss_and_reaches_reference():
    """The stock train() loop drives the walked residual down with no changes."""
    from pinn.config import reference_trajectory
    from pinn.train import predict, train

    cfg = Config(sequential=True, depth=2, width=16, epochs=400, n_collocation=64,
                 lbfgs_iters=0, log_every=100, eval_every=200, print_every=0, seed=0)
    model, history = train(cfg)

    assert history.loss[-1] < 0.05 * history.loss[0]
    t, ref = reference_trajectory(cfg, n=201)
    rmse = np.sqrt(((predict(model, t) - ref) ** 2).mean())
    # Loose bound only. The walk re-integrates on the grid it is handed, so this
    # 201-point read-out is far too coarse to measure the scheme; see
    # test_sequential_walk_has_a_first_order_error_floor.
    assert rmse < 1e-2


def test_sequential_rejects_the_soft_ic():
    """A pinned initial condition makes the soft-IC penalty vacuous."""
    with pytest.raises(ValueError, match="silent no-op"):
        Config(sequential=True, ic="soft")
    assert Config(sequential=True).ic == "hard"


def test_sequential_run_gets_its_own_architecture_tag():
    """So a sequential run cannot overwrite the single-domain run of the same shape."""
    plain, seq = Config(depth=4, width=60), Config(depth=4, width=60, sequential=True)
    assert plain.arch == "4x60" and seq.arch == "4x60_seq"
    assert "sequential walk" in seq.label and "sequential" not in plain.label
    assert plain.label == Config().label        # unchanged for existing runs


# --------------------------------------------------------------------------
# parallel (batched) vs sequential (one point at a time) evaluation
# --------------------------------------------------------------------------
def test_residual_of_a_point_is_independent_of_its_batchmates():
    """The collocation system is diagonal, so batching cannot couple anything.

    The residual at t_i depends only on theta and t_i: the right-hand side f uses
    only that point's own state, and the derivative is autograd of a network whose
    input is the single scalar t_i. Nothing downstream can make point i depend on
    point j, which is the whole reason one batched pass and a loop are the same.
    """
    import torch

    from pinn.pinn import PINN, residual_parts
    from pinn.train import make_grid

    cfg = Config(depth=1, width=8, n_collocation=64)
    model = PINN(cfg)
    grid = make_grid(cfg, torch.device("cpu"))

    full = residual_parts(model, grid.clone().requires_grad_(True)).r.detach()
    picked = torch.randperm(cfg.n_collocation)[:8]
    alone = residual_parts(model, grid[picked].clone().requires_grad_(True)).r.detach()

    assert torch.allclose(alone, full[picked], atol=1e-6)


def test_batched_and_per_point_accumulation_train_identically():
    """Evaluating all N_c points in parallel is exact, not an approximation.

    Trains the same network from the same weights twice: once with a single batched
    backward pass over the whole collocation set, once with a literal Python loop
    that differentiates each point's share of the mean and accumulates it. The loss
    trajectories and the trained weights must agree to float32 round-off.
    """
    import torch

    from pinn.pinn import PINN, residual_parts
    from pinn.train import get_device, make_grid, set_seed

    cfg = Config(depth=1, width=8, n_collocation=64, epochs=3, seed=0)
    assert cfg.ic == "hard"            # the IC term is zero and shared, by design
    device = get_device()
    grid = make_grid(cfg, device)

    trajectories, params = {}, {}
    for mode in ("batched", "looped"):
        set_seed(cfg.seed)
        model = PINN(cfg).to(device)
        adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_start)
        traj = []
        for _ in range(cfg.epochs):
            adam.zero_grad()
            if mode == "batched":
                loss = residual_parts(model, grid.clone().requires_grad_(True)).r.pow(2).mean()
                loss.backward()
                traj.append(float(loss.detach()))
            else:
                acc = 0.0
                for i in range(cfg.n_collocation):
                    r = residual_parts(model, grid[i:i+1].clone().requires_grad_(True)).r
                    share = r.pow(2).sum() / (3 * cfg.n_collocation)
                    share.backward()
                    acc += float(share.detach())
                traj.append(acc)
            adam.step()
        trajectories[mode] = traj
        params[mode] = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()

    batched, looped = np.array(trajectories["batched"]), np.array(trajectories["looped"])
    # Round-off, not error: the relative difference must sit at float32 scale.
    assert (np.abs(batched - looped) / np.abs(batched)).max() < 1e-5
    assert torch.allclose(params["batched"], params["looped"], rtol=1e-3, atol=0)
