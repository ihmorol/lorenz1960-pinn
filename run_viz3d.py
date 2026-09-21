"""Regenerate the extra figures for finished runs: python run_viz3d.py runs/<tag> [...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pinn import viz  # noqa: E402

if __name__ == "__main__":
    for run in sys.argv[1:]:
        for path in viz.generate_run_extras(run):
            print(path)
