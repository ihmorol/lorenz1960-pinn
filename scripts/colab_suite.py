"""One-command Colab driver for the paper's ablation suite.

Runs every arm in order -- R1 (windowed, causal off), R4 (candidate seeds),
R2 (warm start off), R3 (causal single network), R6 (sequential single
network) -- then zips all artifacts to paper_ablations_results.zip for
download from the Colab file panel. Optimizer budgets are the candidate's,
unchanged. Every step is resume-safe: rerun the script after a disconnect
and finished arms are skipped.

Intended runtime on a Colab T4 GPU: roughly 2.5-3.5 hours.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time

STEPS = [
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r1", "--seeds", "1,2,3"],
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r4", "--seeds", "1,2,3,4"],
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r2", "--seeds", "1,2,3"],
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r3", "--seeds", "1,2"],
    [sys.executable, "scripts/sequential_run.py"],
]


def main() -> None:
    import torch
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
          flush=True)
    started = time.perf_counter()
    for cmd in STEPS:
        print(f"\n=== {' '.join(cmd[1:])} ({(time.perf_counter() - started) / 60:.0f} min elapsed) ===",
              flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"step failed: {' '.join(cmd)}")
    shutil.make_archive("paper_ablations_results", "zip", "runs/paper_ablations")
    print(f"\n[done] {(time.perf_counter() - started) / 60:.0f} min -> paper_ablations_results.zip")


if __name__ == "__main__":
    main()
