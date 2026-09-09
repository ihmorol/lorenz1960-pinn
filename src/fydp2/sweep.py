"""Architecture sweep: depth x width, one run each, at a single fixed seed.

Nine configurations -- depth in {3, 4, 5} crossed with width in {50, 60, 70} --
trained under otherwise identical settings and written to ``runs/<depth>x<width>/``.

Because :attr:`Config.seed` drives both the weight initialisation and the Latin
hypercube draw, every architecture sees the *same* 3000 collocation points, so
differences between cells are attributable to the network alone. With one seed
per cell there is no estimate of run-to-run variance, so the resulting table
reports what each architecture achieved on this seed; it does not establish that
one architecture is better than another. Pass ``seeds=(0, 1, 2)`` to
:func:`sweep` if that stronger claim is ever needed.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import figures
from .config import Config
from .history import FLOAT_FORMAT
from .train import main as train_and_save

DEPTHS = (3, 4, 5)
WIDTHS = (50, 60, 70)
SNAPSHOT_EVERY = 50
COMPARISON_METRIC = "rmse_combined_l2"


def sweep_config(base: Config, depth: int, width: int, seed: int | None = None) -> Config:
    """Derive one sweep run's config, pointing every output at ``runs/<arch>/``."""
    cfg = replace(base, depth=depth, width=width, seed=base.seed if seed is None else seed)
    tag = cfg.arch if seed is None else f"{cfg.arch}_seed{seed}"
    root = f"{base.runs_dir}/{tag}"
    return replace(
        cfg,
        results_dir=root,
        ckpt_dir=f"{root}/history",
        snapshot_every=SNAPSHOT_EVERY,
    )


def run_one(cfg: Config) -> pd.DataFrame:
    """Train one architecture and return the summary row it just wrote."""
    train_and_save(cfg)
    return pd.read_csv(cfg.results_path / "run_summary.csv")


def sweep(
    base: Config | None = None,
    depths: tuple[int, ...] = DEPTHS,
    widths: tuple[int, ...] = WIDTHS,
    seeds: tuple[int, ...] = (),
    metric: str = COMPARISON_METRIC,
) -> pd.DataFrame:
    """Run the full grid, write ``runs/comparison.csv`` and the sweep figures.

    ``runs/<arch>/`` is overwritten on rerun: the runs are deterministic at a fixed
    seed, so a rerun reproduces the same bytes. Nothing here touches the
    run-of-record under ``src/fydp2/``.
    """
    base = base or Config()
    combos = list(itertools.product(depths, widths, seeds or (base.seed,)))
    out = base.runs_path
    out.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    started = time.perf_counter()
    for k, (depth, width, seed) in enumerate(combos, start=1):
        cfg = sweep_config(base, depth, width, seed if seeds else None)
        elapsed = time.perf_counter() - started
        print(f"\n=== [{k}/{len(combos)}] {cfg.arch} seed {cfg.seed} "
              f"({elapsed / 60:.1f} min elapsed) -> {cfg.results_path} ===", flush=True)
        rows.append(run_one(cfg))

    comparison = pd.concat(rows, ignore_index=True).sort_values(
        ["depth", "width", "seed"]).reset_index(drop=True)
    comparison.to_csv(out / "comparison.csv", index=False, float_format=FLOAT_FORMAT)

    figures.generate_sweep(comparison, out / "figures", metric)
    print(f"\n[sweep] {len(combos)} runs in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"[sweep] comparison table -> {out / 'comparison.csv'}")
    print(comparison[["arch", "n_params", metric, "final_loss",
                      "residual_generalisation_gap", "wall_clock_s"]].to_string(index=False))
    return comparison


def load_comparison(runs_dir: str | Path = "runs") -> pd.DataFrame:
    return pd.read_csv(Path(runs_dir) / "comparison.csv")


if __name__ == "__main__":
    sweep()
