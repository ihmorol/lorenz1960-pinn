"""Index page for finished runs: python run_index.py [runs/<tag> ...]."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pinn.viz.index import write_index  # noqa: E402

if __name__ == "__main__":
    root = Path("runs")
    runs = [Path(r) for r in sys.argv[1:]] or [p.parent for p in sorted(root.rglob("run_summary.csv"))]
    print(write_index(root, [r.resolve().relative_to(root.resolve()).as_posix() for r in runs]))
