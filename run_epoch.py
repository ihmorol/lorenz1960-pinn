"""Entry point: inspect a single epoch's collocation batch, point by point.

    python run_epoch.py           # 1 epoch
    python run_epoch.py 5         # 5 epochs, one snapshot each

Trains from scratch for the given number of epochs with a snapshot every epoch,
then prints the full N_c-row table for the last one: for each collocation point,
its time, the raw network output, the trial solution, the autograd derivative,
the physics right-hand side, and the residual.

No figures are rendered -- one epoch is too few points to plot -- so this is the
fast way to see exactly what one training step computes.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from pinn.config import Config  # noqa: E402
from pinn.train import train  # noqa: E402

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = replace(Config(), epochs=epochs, snapshot_every=1, log_every=1, eval_every=1,
                  print_every=0, results_dir="runs/epoch_probe",
                  ckpt_dir="runs/epoch_probe/history")
    model, history = train(cfg)
    history.snapshots.finalize(cfg.results_path)

    path = cfg.results_path / "breakdown" / f"epoch_{epochs - 1:06d}.csv"
    frame = pd.read_csv(path)
    pd.set_option("display.width", 200, "display.max_columns", None)
    print(f"\n{path}  ({len(frame)} collocation points x {len(frame.columns)} columns)\n")
    print(frame.head(10).to_string(index=False))
    print(f"\n... {len(frame) - 10} more rows\n")

    # The residual loss is the mean over 3*N_c entries, so each point's share is
    # r_sq/(3*N_c); summed down the table it must reproduce the reported loss.
    print(f"sum of loss_contribution : {frame['loss_contribution'].sum():.12e}")
    print(f"loss reported this epoch : {history.loss[-1]:.12e}")
