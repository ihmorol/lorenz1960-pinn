"""Training film. Render: RUN=<run dir> manim -ql script.py LearningTheLoop DescendingTheSurface"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from manim import (BLUE, BLUE_D, BLUE_E, DEGREES, GREY, RED, UL, YELLOW, Arrow3D, Create, Dot3D,
                   FadeIn, Surface, Text, ThreeDAxes, ThreeDScene, Transform, VGroup, VMobject,
                   interpolate_color)

RUN = Path(os.environ.get("RUN", "runs/4x60_f64_unit"))
BG = "#1C1C1C"


def frames(n_frames=40, n_points=300):
    files = sorted((RUN / "breakdown").glob("epoch_*.csv"))
    pick = np.unique(np.linspace(0, len(files) - 1, n_frames).round().astype(int))
    out = []
    for i in pick:
        f = pd.read_csv(files[i]).sort_values("t")
        out.append(f.iloc[:: max(1, len(f) // n_points)])
    return out


class LearningTheLoop(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        fr = frames()
        ref = fr[0][["ref_x", "ref_y", "ref_z"]].to_numpy()
        scale = 2.0 / np.abs(ref).max()
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.add(ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2]).set_opacity(0.15))
        loop = VMobject(color=GREY, stroke_opacity=0.4).set_points_smoothly([*(ref * scale)])
        self.play(Create(loop), run_time=2)
        self.wait(1)
        counter = Text("epoch 0", font_size=28).to_corner(UL)
        self.add_fixed_in_frame_mobjects(counter)
        curve = None
        self.begin_ambient_camera_rotation(rate=0.05)
        for f in fr:
            pts = f[["x", "y", "z"]].to_numpy() * scale
            err = np.clip((np.log10(f.err_norm.to_numpy() + 1e-9) + 6) / 6, 0, 1)
            new = VGroup(*[Dot3D(p, radius=0.03, color=interpolate_color(BLUE, RED, float(e)))
                           for p, e in zip(pts, err)])
            label = Text(f"epoch {int(f.epoch.iloc[0])}", font_size=28).to_corner(UL)
            self.add_fixed_in_frame_mobjects(label)
            if curve is None:
                curve = new
                self.play(FadeIn(curve), Transform(counter, label), run_time=0.5)
            else:
                self.play(Transform(curve, new), Transform(counter, label), run_time=0.5)
        self.wait(2)


class DescendingTheSurface(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        d = np.load(RUN / "figures" / "loss_landscape.npz")
        a, b, Z, proj, lp = d["a"], d["b"], d["logZ"], d["proj"], d["log_path"]
        sx, sy = 4 / (a[-1] - a[0]), 4 / (b[-1] - b[0])
        z0, sz = Z.min(), 3 / max(Z.max() - Z.min(), 1e-9)

        def height(u, v):
            return (np.interp(u, a, Z[np.abs(b - v).argmin()]) - z0) * sz

        surf = Surface(lambda u, v: np.array([(u - a[0]) * sx - 2, (v - b[0]) * sy - 2, height(u, v)]),
                       u_range=[a[0], a[-1]], v_range=[b[0], b[-1]], resolution=(30, 30),
                       fill_opacity=0.5, checkerboard_colors=[BLUE_E, BLUE_D])
        self.set_camera_orientation(phi=60 * DEGREES, theta=-60 * DEGREES)
        self.play(Create(surf), run_time=2)
        self.wait(1)
        path = [np.array([(p[0] - a[0]) * sx - 2, (p[1] - b[0]) * sy - 2, (l - z0) * sz])
                for p, l in zip(proj, lp)]
        trace = VMobject(color=RED).set_points_as_corners(path[:2])
        dot = Dot3D(path[0], color=YELLOW, radius=0.06)
        self.add(trace, dot)
        for k in range(1, len(path)):
            step = path[k] - path[k - 1]
            if np.linalg.norm(step) > 1e-6:
                arrow = Arrow3D(path[k - 1], path[k - 1] - 3 * step, color=YELLOW, thickness=0.01)
                self.add(arrow)
            trace.add_points_as_corners([path[k]])
            self.play(dot.animate.move_to(path[k]), run_time=0.15)
            if np.linalg.norm(step) > 1e-6:
                self.remove(arrow)
        self.wait(2)
