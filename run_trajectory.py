"""Entry point: watch the whole collocation batch move, epoch by epoch.

Two modes.

From a finished run — no retraining, reads the 3000-row breakdown CSVs that the
standard snapshot writer already wrote:

    python run_trajectory.py --run runs/3x50

From a fresh short run, if you want a denser cadence than the finished runs have:

    python run_trajectory.py --train --epochs 2000 --every 1

Both write their figures to ``<run dir>/trajectory/``. ``--animate`` adds the
GIF; ``--experiment`` adds the batched-versus-iterated comparison, which is the
part that speaks to "should the 3000 points be trained one at a time".

The training code is not involved: ``--train`` only sets ``Config.snapshot_every``
and calls the stock ``train()``. Nothing here writes outside the directory it
owns, so the run-of-record pipeline is untouched.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402

from pinn import figures, trajectory, trajectory_figures as tfig  # noqa: E402
from pinn.config import Config  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--run", type=Path, default=Path("runs/3x50"),
                        help="run directory holding breakdown/epoch_*.csv (default: runs/3x50)")
    source.add_argument("--train", action="store_true",
                        help="train a short run instead of reading a finished one")
    p.add_argument("--epochs", type=int, default=2000, help="epochs when training (default: 2000)")
    p.add_argument("--every", type=int, default=1,
                   help="epochs per snapshot when training (default: 1, i.e. every epoch; "
                        "each snapshot is a ~1.3 MB CSV, so 2000 epochs at 1 writes ~2.6 GB)")
    p.add_argument("--train-dir", type=Path, default=None,
                   help="where --train writes its run (default: runs/traj_<arch>); it is "
                        "deliberately not src/pinn/results, which belongs to the run of record")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--n-collocation", type=int, default=3000)
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: <run dir>/trajectory)")
    p.add_argument("--frames", type=int, default=90,
                   help="animation frames (default: 90); every frame is a snapshot, so more frames means a longer GIF")
    p.add_argument("--max-frames", type=int, default=0,
                   help="cap on snapshots read from disk (0 = every snapshot found, the default)")
    p.add_argument("--max-lines", type=int, default=100, help="epochs drawn per 2-D panel")
    p.add_argument("--panels", type=int, default=6, help="3-D panel count")
    p.add_argument("--thin", type=int, default=4, help="keep every Nth collocation point in dense figures")
    p.add_argument("--animate", action="store_true", help="also write the GIF")
    p.add_argument("--experiment", action="store_true",
                   help="also run the batched-vs-iterated equivalence and batch-size comparison")
    p.add_argument("--point-budget", type=int, default=3_000_000,
                   help="point evaluations per arm in the batch-size comparison")
    p.add_argument("--max-wall-s", type=float, default=240.0,
                   help="per-arm wall-clock cap for the batch-size comparison")
    return p.parse_args(argv)


def load_source(args: argparse.Namespace) -> tuple[trajectory.TrajectoryData, Path]:
    if args.train:
        cfg = Config(depth=args.depth, width=args.width, n_collocation=args.n_collocation,
                     epochs=args.epochs, print_every=max(1, args.epochs // 10))
        # Never train into src/pinn/results: that is the run of record, and a
        # dense cadence would bury it in breakdown CSVs.
        train_dir = args.train_dir or (cfg.runs_path / f"traj_{cfg.arch}")
        cfg = replace(cfg, snapshot_every=max(1, args.every),
                      results_dir=str(train_dir), ckpt_dir=str(train_dir / "history"))
        print(f"[traj] training {cfg.label} for {cfg.epochs} epochs, "
              f"a snapshot every {cfg.snapshot_every} -> {cfg.results_path / 'breakdown'}",
              flush=True)
        _, _, data = trajectory.record_per_epoch(cfg, every=cfg.snapshot_every)
        return data, cfg.results_path

    breakdown = args.run / "breakdown"
    data = trajectory.load_from_breakdown(
        breakdown, max_frames=args.max_frames, label=args.run.name)
    print(f"[traj] {data.n_frames} snapshots x {data.n_points} collocation points "
          f"from {breakdown}", flush=True)
    return data, args.run


def render(data: trajectory.TrajectoryData, out: Path, args: argparse.Namespace) -> dict[str, list[Path]]:
    figures.set_style()
    fig_dir = out / "figures"
    rmse = data.rmse()
    common = dict(epochs=data.epochs, pred=data.pred, ref=data.ref)
    # The box in the full-run panels has to cover the early epochs' overshoot, so
    # the converged orbit ends up a squashed line inside it. These panels zoom to
    # the tail of training instead, where the axes can close in on the orbit shape.
    tail = np.linspace(int(0.8 * data.n_frames), data.n_frames - 1, args.panels).round().astype(int)

    written = {
        "trajectory_3d_panels": figures.save_figure(
            tfig.fig_trajectory_3d_panels(n_panels=args.panels, mode="line",
                                          rmse=rmse, label=data.label, **common),
            fig_dir, "trajectory_3d_panels"),
        "trajectory_3d_final": figures.save_figure(
            tfig.fig_trajectory_3d_panels(n_panels=args.panels, mode="line", rmse=rmse,
                                          label=data.label, picks=np.unique(tail), **common),
            fig_dir, "trajectory_3d_final"),
        "trajectory_3d_points": figures.save_figure(
            tfig.fig_trajectory_3d_panels(n_panels=args.panels, mode="scatter",
                                          rmse=rmse, label=data.label, **common),
            fig_dir, "trajectory_3d_points"),
        "trajectory_projections": figures.save_figure(
            tfig.fig_trajectory_projections(max_lines=args.max_lines, thin=args.thin,
                                            label=data.label, **common),
            fig_dir, "trajectory_projections"),
        "trajectory_projections_true_aspect": figures.save_figure(
            tfig.fig_trajectory_projections(max_lines=args.max_lines, thin=args.thin,
                                            equal_aspect=True, label=data.label, **common),
            fig_dir, "trajectory_projections_true_aspect"),
        "trajectory_time_slices": figures.save_figure(
            tfig.fig_trajectory_time_slices(t=data.t, max_lines=args.max_lines,
                                            thin=args.thin, label=data.label, **common),
            fig_dir, "trajectory_time_slices"),
        "trajectory_extent": figures.save_figure(
            tfig.fig_trajectory_extent(log_epoch=True, label=data.label, **common),
            fig_dir, "trajectory_extent"),
    }

    if args.animate:
        gif = tfig.animate_trajectory(
            epochs=data.epochs, t=data.t, pred=data.pred, ref=data.ref,
            path=fig_dir / "trajectory_animation.gif", n_frames=args.frames,
            thin=args.thin, label=data.label)
        written["trajectory_animation"] = [gif]
    return written


def run_experiments(cfg: Config, out: Path, point_budget: int, max_wall_s: float) -> dict[str, list[Path]]:
    """The two measurements behind the "should we iterate the 3000 points" question."""
    print(f"[traj] batching equivalence on {cfg.n_collocation} points "
          f"({cfg.arch}) ...", flush=True)
    report = trajectory.gradient_equivalence(cfg)
    print(f"[traj]   batched vs point-by-point: loss differs by "
          f"{report['rel_loss_diff']:.2e}, gradient by {report['rel_grad_diff']:.2e}, "
          f"cosine {report['grad_cosine']:.9f}", flush=True)

    print(f"[traj] batch-size comparison, {point_budget:,} point evaluations per arm, "
          f"{max_wall_s:.0f}s cap ...", flush=True)
    curves, timing = trajectory.batch_size_comparison(
        cfg, point_budget=point_budget, max_wall_s=max_wall_s)
    written = trajectory.write_experiment_tables(report, curves, timing, out / "tables")
    for key, seconds in timing.items():
        rows = curves[curves["batch_size"] == key]
        print(f"[traj]   batch {key:>4}: {rows['step'].iloc[-1] + 1:,} steps, "
              f"{rows['loss'].iloc[-1]:.3e} final loss, {seconds:.1f}s", flush=True)

    figures.set_style()
    written["batching_equivalence_figure"] = figures.save_figure(
        tfig.fig_gradient_equivalence(report, cfg.label), out / "figures", "batching_equivalence")
    written["batching_cost_figure"] = figures.save_figure(
        tfig.fig_batch_size_comparison(curves, timing, cfg.label), out / "figures", "batching_cost")
    return written


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    data, run_dir = load_source(args)
    out = args.out or (run_dir / "trajectory")
    out.mkdir(parents=True, exist_ok=True)

    npz = trajectory.save_npz(data, out / "trajectory.npz")
    trajectory.trajectory_frame(data).to_csv(out / "trajectory_epochs.csv", index=False)
    print(f"[traj] saved {data.n_frames} frames x {data.n_points} points -> {npz.name}", flush=True)

    written = render(data, out, args)
    if args.experiment:
        cfg = Config(depth=args.depth, width=args.width, n_collocation=args.n_collocation)
        written.update(run_experiments(cfg, out, args.point_budget, args.max_wall_s))

    total = sum(len(v) for v in written.values())
    print(f"[traj] wrote {total} files under {out} in {time.perf_counter() - t0:.1f}s", flush=True)
    for name, paths in written.items():
        print(f"       {name}: {', '.join(p.name for p in paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
