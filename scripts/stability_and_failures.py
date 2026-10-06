"""Multi-seed stability and failure-case training runs for the conference study.

Family 1 (stability): one preset configuration retrained under several seeds;
the table reports the run-to-run spread of the final error.

Family 2 (failures): configurations chosen to break plain single-window PINN
training -- long horizon, large initial amplitude, high wavenumbers, the
degenerate k=l case -- each trained twice, plain and with the causal-windowed
arm that is supposed to rescue it.

Every run uses sweep.run_one's resume semantics, so re-invoking the script
continues an interrupted batch instead of redoing finished cells.

Usage:
    python scripts/stability_and_failures.py --stability --preset default --seeds 0,1,2,3,4
    python scripts/stability_and_failures.py --failures
    both flags accept --epochs N to shrink every run (smoke testing only)
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import config_for, run_one  # noqa: E402

CAUSAL = dict(causal_eps_schedule=(0.01, 0.1, 1.0, 10.0), causal_delta=0.99,
              causal_max_iters=2000, n_windows=9)

FAILURES = {
    "long_horizon": lambda c: replace(c, t_span=(0.0, 5.0)),
    "large_ic": lambda c: replace(c, initial_state=(4.0, 4.0, 4.0)),
    "high_wavenumber": lambda c: replace(c, k=4.0, l=0.5),
    "degenerate_kl": lambda c: replace(c, k=1.0, l=1.0),
}


def preset(name: str) -> Config:
    if name == "candidate":
        return config_for(_REPO / "runs" / "4x60_f64_unit_win27_causal_warm")
    return Config()


def _dirs(tag: str) -> dict:
    return dict(results_dir=f"runs/stability/{tag}", ckpt_dir=f"runs/stability/{tag}/history")


def stability(base: Config, seeds: tuple[int, ...], epochs: int | None) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        cfg = replace(base, seed=seed, **(_dirs(f"{base.arch}_seed{seed}")))
        if epochs:
            cfg = replace(cfg, epochs=epochs)
        print(f"\n=== stability: {cfg.arch} seed {seed} ===", flush=True)
        row = run_one(cfg, resume=True).iloc[0]
        rows.append({"preset": base.arch, "seed": seed,
                     "final_loss": row["final_loss"], "rmse_combined_l2": row["rmse_combined_l2"],
                     "wall_clock_s": row["wall_clock_s"]})
    return pd.DataFrame(rows)


def failures(base: Config, epochs: int | None) -> pd.DataFrame:
    rows = []
    for name, mutate in FAILURES.items():
        for arm, extra in (("plain", {}), ("causal_windowed", CAUSAL)):
            cfg = replace(mutate(base), **extra, **(_dirs(f"{name}_{arm}")))
            if epochs:
                cfg = replace(cfg, epochs=epochs, causal_max_iters=min(cfg.causal_max_iters, epochs))
            print(f"\n=== failure: {name} / {arm} ===", flush=True)
            row = run_one(cfg, resume=True).iloc[0]
            rows.append({"case": name, "arm": arm, "arch": cfg.arch,
                         "final_loss": row["final_loss"], "rmse_combined_l2": row["rmse_combined_l2"],
                         "wall_clock_s": row["wall_clock_s"]})
            pd.DataFrame(rows).to_csv(_REPO / "runs" / "stability" / "failures.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--preset", choices=("default", "candidate"), default="default")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    if not args.stability and not args.failures:
        ap.error("pick --stability, --failures, or both")

    base = preset(args.preset)
    out = _REPO / "runs" / "stability"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if args.stability:
        table = stability(base, tuple(int(s) for s in args.seeds.split(",")), args.epochs)
        table.to_csv(out / "stability.csv", index=False)
        print("\n[stability]\n", table.to_string(index=False))
        spread = table.rmse_combined_l2.agg(["mean", "std"])
        print(f"rmse mean={spread['mean']:.3e} std={spread['std']:.3e}")
    if args.failures:
        table = failures(base, args.epochs)
        table.to_csv(out / "failures.csv", index=False)
        print("\n[failures]\n", table.to_string(index=False))
    print(f"\n[done] {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
