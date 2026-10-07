"""One-command Colab driver for the paper's ablation suite.

Runs every arm in order -- R1 (windowed, causal off), R4 (candidate seeds),
R2 (warm start off), R3 (causal single network), R6 (sequential single
network) -- with results persisted to Google Drive after each arm, so a
runtime disconnect loses at most the arm in flight. Optimizer budgets are
the candidate's, unchanged. Every step is resume-safe: rerun the script
after a reconnect and finished arms are skipped.

Training happens on Colab's local disk (fast); the runs directory is copied
to /content/drive/MyDrive/lorenz1960-pinn-ablation/ after each arm and
restored from there at startup. A zip of all artifacts is left both in the
working directory and on Drive.

Intended runtime on a Colab T4 GPU: roughly 2.5-3.5 hours.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/lorenz1960-pinn-ablation")
RUNS = Path("runs/paper_ablations")

STEPS = [
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r1", "--seeds", "1,2,3"],
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r4", "--seeds", "1,2,3,4"],
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r2", "--seeds", "1,2,3"],
    [sys.executable, "scripts/paper_ablations.py", "--arms", "r3", "--seeds", "1,2"],
    [sys.executable, "scripts/sequential_run.py"],
]


def sync_to_drive() -> None:
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUNS, DRIVE_ROOT / "paper_ablations", dirs_exist_ok=True)
    print("[drive] results saved", flush=True)


def main() -> None:
    import torch
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
          flush=True)

    on_colab = "google.colab" in sys.modules or _importable("google.colab")
    if on_colab:
        from google.colab import drive
        drive.mount("/content/drive")
        if (DRIVE_ROOT / "paper_ablations").exists():
            shutil.copytree(DRIVE_ROOT / "paper_ablations", RUNS, dirs_exist_ok=True)
            print("[drive] previous results restored", flush=True)

    started = time.perf_counter()
    try:
        for cmd in STEPS:
            print(f"\n=== {' '.join(cmd[1:])} "
                  f"({(time.perf_counter() - started) / 60:.0f} min elapsed) ===", flush=True)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.exit(f"step failed: {' '.join(cmd)}")
            if on_colab:
                sync_to_drive()
    finally:
        if on_colab and RUNS.exists():
            sync_to_drive()
        if RUNS.exists():
            shutil.make_archive("paper_ablations_results", "zip", RUNS)
            if on_colab:
                shutil.copy2("paper_ablations_results.zip", DRIVE_ROOT)
            print(f"\n[done] {(time.perf_counter() - started) / 60:.0f} min "
                  f"-> paper_ablations_results.zip", flush=True)


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
