"""Entry point: train the Lorenz-1960 PINN by time marching instead of all at once.

    python run_marching.py --segments 8 --epochs-per-stage 2500

Splits ``[0, 1]`` into stages, trains them in order, and hands each stage the
previous stage's endpoint as its initial condition. Compare the resulting row of
``marching_summary.csv`` against ``runs/<arch>/run_summary.csv``: the two are the
same problem with the same total epoch budget, so the difference is the scheme.

The single-domain baseline is the architecture sweep's own output, which already
exists on disk:

    python run_marching.py --segments 8 --epochs-per-stage 2500 --compare runs/4x60

Nothing here modifies the training loop, the config, or the figure suite; it
imports them.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from pinn import figures, marching, trajectory_figures as tfig  # noqa: E402
from pinn.config import reference_trajectory  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", type=int, default=8, help="number of stages (default: 8)")
    p.add_argument("--epochs-per-stage", type=int, default=2500,
                   help="Adam epochs per stage (default: 2500; 8 stages x 2500 matches "
                        "the sweep's 20000-epoch budget)")
    p.add_argument("--finetune-epochs", type=int, default=0,
                   help="joint epochs at the end with trainable anchors (default: 0, "
                        "i.e. strictly sequential)")
    p.add_argument("--lbfgs-iters", type=int, default=0, help="L-BFGS iterations per stage")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--n-collocation", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: runs/marching_<segments>x<arch>)")
    p.add_argument("--compare", type=Path, default=None,
                   help="a finished single-domain run directory to diff the summary against")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    cfg = marching.MarchingConfig(
        depth=args.depth, width=args.width, n_collocation=args.n_collocation,
        seed=args.seed, n_segments=args.segments, epochs=args.epochs_per_stage,
        finetune_epochs=args.finetune_epochs, lbfgs_iters=args.lbfgs_iters,
        snapshot_every=0, print_every=max(1, args.epochs_per_stage // 4),
    )
    out = args.out or (cfg.runs_path / f"marching_{args.segments}x{cfg.arch}")
    print(f"[march] {cfg.label} | {cfg.n_segments} stages x {cfg.epochs} epochs "
          f"= {cfg.total_epochs} total -> {out}", flush=True)

    model, history, frames = marching.train_marching(cfg)
    summary = marching.marching_summary(model, history, frames, cfg)
    written = marching.write_marching_tables(frames, summary, out)

    print(frames["stages"][["stage", "t_start", "t_end", "final_loss", "stage_rmse"]]
          .to_string(index=False))
    print(f"\n[march] max boundary jump {summary['max_boundary_jump'].iloc[0]:.3e} "
          f"(the piecewise solution's own discontinuity)", flush=True)
    print(f"[march] {cfg.total_epochs} epochs in {history.wall_clock_s:.1f}s", flush=True)

    figures.set_style()
    t_ref, ref = reference_trajectory(cfg, n=1001)
    pred = marching.predict_marching(model, t_ref)
    paths = figures.save_figure(
        tfig.fig_marching_seams(t_ref, pred, ref, cfg.edges(),
                                stage_rmse=frames["stages"]["stage_rmse"].to_numpy(),
                                label=cfg.label),
        out / "figures", "marching_seams")
    written += paths

    table = summary
    if args.compare is not None:
        baseline = args.compare / "run_summary.csv"
        if baseline.exists():
            single = pd.read_csv(baseline)
            single.insert(0, "scheme", "single-domain")
            single["n_segments"] = 1
            shared = [c for c in summary.columns if c in single.columns]
            table = pd.concat([single[shared], summary[shared]], ignore_index=True)
            table.to_csv(out / "vs_single_domain.csv", index=False)
            written.append(out / "vs_single_domain.csv")
        else:
            print(f"[march] no run_summary.csv under {args.compare}; skipping comparison",
                  flush=True)

    show = ["scheme", "n_segments", "epochs", "wall_clock_s", "final_loss",
            "rmse_combined_l2", "max_abs_error_combined_l2", "max_boundary_jump",
            "residual_mse_dense"]
    print()
    print(table[[c for c in show if c in table.columns]].to_string(index=False))
    print(f"\n[march] wrote {len(written)} files under {out} "
          f"in {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
