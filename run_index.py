"""Index page for finished runs: python run_index.py runs/<tag> [...]  -> runs/index.html"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pinn.viz.index import write_index  # noqa: E402

if __name__ == "__main__":
    runs = [Path(r) for r in sys.argv[1:]]
    print(write_index(runs[0].parent, [r.name for r in runs]))
