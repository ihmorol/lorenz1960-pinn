"""Entry point: train the Lorenz-1960 PINN, evaluate it, and write results.

Run from the repository root:  python run_pinn.py
Outputs go to src/fydp2/results/ (tracked) and data/fydp2/ (gitignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fydp2.train import main  # noqa: E402

if __name__ == "__main__":
    main()
