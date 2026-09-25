"""python run_film.py runs/<tag> [-qh]   -> <run>/film/final.mp4"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    run = Path(sys.argv[1])
    quality = sys.argv[2] if len(sys.argv) > 2 else "-ql"
    out = run / "film"
    out.mkdir(exist_ok=True)
    script = Path(__file__).resolve().parent / "src" / "pinn" / "viz" / "film" / "script.py"
    scenes = ["LearningTheLoop"]
    if (run / "figures" / "loss_landscape.npz").exists():
        scenes.append("DescendingTheSurface")
    trail = run / "history" / "param_trail.npz"
    if trail.exists() and "weights" in np.load(trail):
        scenes.append("TheCausalFront")
    env = {**os.environ, "RUN": str(run.resolve())}
    subprocess.run(["manim", quality, "--media_dir", str(out), str(script), *scenes], env=env, check=True)
    clips = [next(out.rglob(f"{s}.mp4")) for s in scenes]
    (out / "concat.txt").write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(out / "concat.txt"), "-c", "copy", str(out / "final.mp4")], check=True)
    print(out / "final.mp4")
