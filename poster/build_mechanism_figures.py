#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd"
ASSETS = ROOT / "poster/assets"

NAVY = "#17344d"
TEAL = "#087f83"
AMBER = "#b97020"
PALE = "#eef5f8"
INK = "#263a4c"
GRID = "#c9d9e3"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stage_traces() -> dict[int, tuple[list[int], list[float], bool]]:
    causal = [float(row["min_w"]) for row in read_csv(RUN / "history/causal.csv")]
    stages = read_csv(RUN / "history/stages.csv")
    selected = {0, 10, 26}
    traces: dict[int, tuple[list[int], list[float], bool]] = {}
    offset = 0
    for row in stages:
        count = int(row["adam_steps"])
        values = causal[offset : offset + count]
        if int(row["window"]) in selected and float(row["eps"]) == 10.0:
            traces[int(row["window"])] = (
                list(range(1, count + 1)),
                values,
                row["threshold_met"].lower() == "true",
            )
        offset += count
    assert offset == len(causal), "stage counts must cover the complete causal trace"
    assert traces.keys() == selected, "all selected epsilon=10 stages must be present"
    return traces


def build_weights() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.2,
            "svg.fonttype": "none",
        }
    )
    traces = stage_traces()
    fig, ax = plt.subplots(figsize=(10.8, 4.6), constrained_layout=True)
    colors = {0: TEAL, 10: AMBER, 26: NAVY}
    labels = {
        0: "window 1 · passed",
        10: "window 11 · capped",
        26: "window 27 · passed",
    }
    for window in (0, 10, 26):
        steps, values, met = traces[window]
        ax.plot(steps, values, color=colors[window], linewidth=2.5, label=labels[window])
        ax.scatter(steps[-1], values[-1], s=65, color=colors[window], edgecolor="white", linewidth=1.2, zorder=4)
        if met:
            assert values[-1] > 0.99
        else:
            assert len(values) == 4000 and values[-1] <= 0.99
    ax.axhline(0.99, color="#6b7f8f", linestyle=(0, (6, 4)), linewidth=1.8, label="advance threshold · 0.99")
    ax.set_xlim(0, 4050)
    ax.set_ylim(0.68, 1.012)
    ax.set_xlabel("Adam step within the ε = 10 stage")
    ax.set_ylabel("minimum temporal weight, min w")
    ax.grid(axis="y", color=GRID, alpha=0.65, linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", ncol=2, frameon=False)
    fig.savefig(ASSETS / "defense_weights.svg", transparent=True, metadata={"Description": "Exact min_w traces from the saved candidate-1536pt causal.csv and stages.csv."})
    plt.close(fig)


def build_handoff() -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 510" role="img" aria-labelledby="title desc">
  <title id="title">Measured window handoff behavior</title>
  <desc id="desc">The next window is anchored to the previous endpoint, giving value continuity at the plotting floor, while derivative continuity is not constrained and the saved finite-difference diagnostic is of order ten to the minus three.</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{TEAL}"/></marker>
  </defs>
  <rect width="1200" height="510" rx="28" fill="#ffffff"/>
  <text x="52" y="62" fill="{NAVY}" font-family="DejaVu Sans, Arial, sans-serif" font-size="34" font-weight="700">What crosses each window interface?</text>
  <text x="52" y="98" fill="#526b7e" font-family="DejaVu Sans, Arial, sans-serif" font-size="21">Saved candidate-1536pt diagnostic · 26 interfaces</text>

  <rect x="52" y="140" width="315" height="164" rx="20" fill="{PALE}" stroke="{GRID}" stroke-width="2"/>
  <text x="82" y="192" fill="{TEAL}" font-family="DejaVu Sans, Arial, sans-serif" font-size="23" font-weight="700">WINDOW k</text>
  <text x="82" y="238" fill="{NAVY}" font-family="DejaVu Sans, Arial, sans-serif" font-size="29" font-weight="700">endpoint state</text>
  <text x="82" y="276" fill="{INK}" font-family="DejaVu Sans, Arial, sans-serif" font-size="24">uₖ(tₖ₊₁)</text>

  <path d="M 371 222 C 426 222, 434 222, 486 222" fill="none" stroke="{TEAL}" stroke-width="6" marker-end="url(#arrow)"/>
  <text x="388" y="193" fill="{TEAL}" font-family="DejaVu Sans, Arial, sans-serif" font-size="20" font-weight="700">ANCHOR</text>

  <rect x="493" y="140" width="315" height="164" rx="20" fill="#eaf7f6" stroke="#a3d5d2" stroke-width="3"/>
  <text x="523" y="192" fill="{TEAL}" font-family="DejaVu Sans, Arial, sans-serif" font-size="23" font-weight="700">WINDOW k + 1</text>
  <text x="523" y="238" fill="{NAVY}" font-family="DejaVu Sans, Arial, sans-serif" font-size="29" font-weight="700">hard initial state</text>
  <text x="523" y="276" fill="{INK}" font-family="DejaVu Sans, Arial, sans-serif" font-size="24">uₖ₊₁(tₖ₊₁) = uₖ(tₖ₊₁)</text>

  <rect x="840" y="140" width="308" height="164" rx="20" fill="#fff6e8" stroke="#e9c898" stroke-width="3"/>
  <text x="870" y="192" fill="{AMBER}" font-family="DejaVu Sans, Arial, sans-serif" font-size="23" font-weight="700">NOT CONSTRAINED</text>
  <text x="870" y="238" fill="{NAVY}" font-family="DejaVu Sans, Arial, sans-serif" font-size="28" font-weight="700">interface slope</text>
  <text x="870" y="276" fill="{INK}" font-family="DejaVu Sans, Arial, sans-serif" font-size="24">duₖ/dt ≠ duₖ₊₁/dt</text>

  <rect x="52" y="344" width="526" height="112" rx="18" fill="#eaf7f6"/>
  <text x="82" y="386" fill="{TEAL}" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="700">STATE-VALUE MISMATCH</text>
  <text x="82" y="430" fill="{NAVY}" font-family="DejaVu Sans, Arial, sans-serif" font-size="30" font-weight="700">at the 10⁻¹⁶ plotting floor</text>

  <rect x="604" y="344" width="544" height="112" rx="18" fill="#fff6e8"/>
  <text x="634" y="386" fill="{AMBER}" font-family="DejaVu Sans, Arial, sans-serif" font-size="22" font-weight="700">FINITE-DIFFERENCE SLOPE MISMATCH</text>
  <text x="634" y="430" fill="{NAVY}" font-family="DejaVu Sans, Arial, sans-serif" font-size="30" font-weight="700">remains on the order of 10⁻³</text>
</svg>'''
    (ASSETS / "defense_handoff.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_weights()
    build_handoff()
