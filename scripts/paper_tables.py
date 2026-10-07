"""Generate the revised paper's result tables as markdown and LaTeX booktabs.

Reads the live ablation log and the run-of-record summaries, emits
docs/PAPER_TABLES.md, and (with --latex) prints booktabs LaTeX to stdout.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(r"E:\University\FYDP\lorenz1960-pinn")
ABLATIONS_CSV = MAIN_ROOT / "runs" / "paper_ablations" / "ablations.csv"
FULL_INTERVAL_SUMMARY = MAIN_ROOT / "runs" / "4x60_f64_unit" / "run_summary.csv"
CANDIDATE_SUMMARY = (
    MAIN_ROOT
    / "runs"
    / "causal-window"
    / "candidate-1536pt"
    / "4x60_f64_unit_win27_causal_warm_f1cbeebd"
    / "run_summary.csv"
)
SEQUENTIAL_METRICS = (
    MAIN_ROOT / "runs" / "paper_ablations" / "sequential_single_net" / "error_metrics.csv"
)
OUTPUT_MD = WORKTREE_ROOT / "docs" / "PAPER_TABLES.md"

ABLATION_COLS = [
    "arm",
    "seed",
    "arch",
    "final_loss",
    "rmse_combined_l2",
    "max_abs_error",
    "wall_clock_s",
]
NUMERIC_COLS = ABLATION_COLS[3:]

ISOLATION = {
    "r1_no_causal": "causal weighting removed (windowed + warm start kept)",
    "r2_no_warm": "warm start removed (causal + windowing kept)",
    "r4_candidate": "full method: causal weighting + warm start, windowed",
    "r3_causal_single": "causal weighting on a single network (no windowing)",
    "r5_param_matched": "parameter-matched control",
}
ARM_ORDER = list(ISOLATION)

KNOWN_GOOD = [
    (
        "ablations.csv r1_no_causal_seed1 rmse_combined_l2",
        ABLATIONS_CSV,
        "rmse_combined_l2",
        "arm == 'r1_no_causal_seed1'",
        0.000175485348,
    ),
    (
        "candidate run_summary rmse_combined_l2",
        CANDIDATE_SUMMARY,
        "rmse_combined_l2",
        None,
        6.96696531e-05,
    ),
    (
        "full-interval run_summary rmse_combined_l2",
        FULL_INTERVAL_SUMMARY,
        "rmse_combined_l2",
        None,
        2.23706051,
    ),
]


def sig3(x: float) -> float:
    return float(f"{x:.3g}")


def fmt(x, spec: str = ".3g", missing: str = "--") -> str:
    if x is None:
        return missing
    try:
        if isinstance(x, float) and math.isnan(x):
            return missing
    except TypeError:
        return missing
    return format(x, spec)


def load_ablations() -> pd.DataFrame:
    if not ABLATIONS_CSV.exists():
        return pd.DataFrame(columns=ABLATION_COLS)
    df = pd.read_csv(ABLATIONS_CSV)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return {k: (None if pd.isna(v) else v) for k, v in df.iloc[0].items()}


def mean_pm_std(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "--"
    if len(series) == 1:
        return f"{fmt(series.iloc[0])} (n=1, no std)"
    mean = series.mean()
    std = series.std(ddof=1)
    return f"{fmt(mean)} +/- {fmt(std)}"


def mean_or_missing(series: pd.Series) -> str:
    series = series.dropna()
    return fmt(series.mean()) if not series.empty else "--"


def budget_from_summary(s: dict | None) -> str:
    if s is None:
        return "--"
    epochs, lbfgs = s.get("epochs"), s.get("lbfgs_iters")
    if epochs is None or lbfgs is None:
        return "--"
    return f"{int(epochs):,} / {int(lbfgs):,}"


def ablation_budget(abl: pd.DataFrame, prefix: str) -> str:
    rows = abl[abl["arm"].str.startswith(prefix, na=False)]
    if rows.empty:
        return "--"
    seed_dirs = sorted((ABLATIONS_CSV.parent).glob(f"{rows['arm'].iloc[0]}"))
    for d in seed_dirs:
        s = read_summary(d / "run_summary.csv")
        if s is not None:
            return budget_from_summary(s)
    return "--"


def build_table1(abl: pd.DataFrame, full: dict | None, cand: dict | None) -> list[list[str]]:
    r1 = abl[abl["arm"].str.startswith("r1_no_causal", na=False)]
    f4_rmse = mean_pm_std(r1["rmse_combined_l2"]) if not r1.empty else "--"
    f4_n = int(r1["seed"].nunique()) if not r1.empty else 0
    f4_max = mean_or_missing(r1["max_abs_error"]) if not r1.empty else "--"
    f4_budget = ablation_budget(abl, "r1_no_causal")

    f3_rmse = "--"
    if SEQUENTIAL_METRICS.exists():
        seq = pd.read_csv(SEQUENTIAL_METRICS)
        row = seq[seq.iloc[:, 0].astype(str).str.contains("combined_l2", case=False, na=False)]
        if not row.empty:
            f3_rmse = fmt(float(row.iloc[0, -1]))
        else:
            f3_rmse = "see " + str(SEQUENTIAL_METRICS)
        f3_verdict = "sequential single network"
    else:
        f3_verdict = "PENDING (error_metrics.csv not yet written)"

    return [
        [
            "F1",
            "single network, full interval",
            fmt(full.get("n_collocation") if full else None, ",.0f"),
            budget_from_summary(full),
            fmt(full.get("rmse_combined_l2") if full else None),
            "fails (plateau)",
        ],
        [
            "F2",
            "plateau of F1 (see runs/4x60_f64_unit/figures/error_vs_t.png)",
            "--",
            "--",
            "--",
            "see figure",
        ],
        [
            "F3",
            "sequential single network",
            "--",
            "--",
            f3_rmse,
            f3_verdict,
        ],
        [
            "F4",
            f"windowed-27 without causal (mean +/- std over r1_no_causal_*, n={f4_n})",
            "--",
            f4_budget,
            f4_rmse,
            "max abs err (mean): " + f4_max,
        ],
        [
            "F5",
            "windowed-27 + causal + warm start (rescue)",
            fmt(cand.get("n_collocation") if cand else None, ",.0f"),
            budget_from_summary(cand),
            fmt(cand.get("rmse_combined_l2") if cand else None),
            "succeeds (emphasized rescue)",
        ],
    ]


def build_table2(abl: pd.DataFrame, cand: dict | None) -> list[list[str]]:
    rows = []
    for prefix in ARM_ORDER:
        sub = abl[abl["arm"].str.startswith(prefix, na=False)]
        n = int(sub["seed"].nunique()) if not sub.empty else 0
        rows.append(
            [
                prefix,
                ISOLATION[prefix],
                str(n) if n else "PENDING",
                mean_pm_std(sub["rmse_combined_l2"]) if n else "--",
                mean_or_missing(sub["max_abs_error"]) if n else "--",
                mean_or_missing(sub["wall_clock_s"]) if n else "--",
            ]
        )
    note_rmse = fmt(cand.get("rmse_combined_l2")) if cand else "--"
    note_wall = fmt(cand.get("wall_clock_s"), ".1f") if cand else "--"
    rows.append(
        [
            "note",
            "r4_candidate seed-0 run of record (4x60_f64_unit_win27_causal_warm_f1cbeebd)",
            "0",
            note_rmse,
            fmt(cand.get("max_abs_error_combined_l2")) if cand else "--",
            note_wall,
        ]
    )
    return rows


def build_table3(abl: pd.DataFrame) -> list[list[str]]:
    rows = []
    for prefix in ["r4_candidate", "r1_no_causal"]:
        sub = abl[abl["arm"].str.startswith(prefix, na=False)].sort_values("seed")
        if sub.empty:
            rows.append([prefix, "PENDING", "--", "--", "--", "--", "--"])
            continue
        for _, r in sub.iterrows():
            rows.append(
                [
                    prefix,
                    fmt(r.get("seed"), "d"),
                    str(r.get("arch", "--")),
                    fmt(r.get("final_loss")),
                    fmt(r.get("rmse_combined_l2")),
                    fmt(r.get("max_abs_error")),
                    fmt(r.get("wall_clock_s"), ".1f"),
                ]
            )
    return rows


TABLE_SPECS = [
    (
        "T1: Failed runs and the rescue (RMSE is combined L2; wall clock in seconds)",
        ["run", "description", "training points", "Adam / L-BFGS", "RMSE", "verdict"],
    ),
    (
        "T2: Ablations (RMSE and max error are combined L2; wall clock in seconds)",
        ["arm", "what it isolates", "seeds n", "RMSE mean +/- std", "max error mean", "wall clock mean (s)"],
    ),
    (
        "T3: Per-seed raw results",
        ["arm", "seed", "arch", "final_loss", "rmse_combined_l2", "max_abs_error", "wall_clock_s"],
    ),
]


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def render_markdown(tables: dict[str, list[list[str]]], snapshot_note: str, validation: list[str]) -> str:
    lines = ["# Paper tables", "", snapshot_note, ""]
    for title, header in TABLE_SPECS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in tables[title]:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.extend(f"- {v}" for v in validation)
    lines.append("")
    return "\n".join(lines)


def render_latex(tables: dict[str, list[list[str]]]) -> str:
    out = []
    for title, header in TABLE_SPECS:
        out.append(f"% {title}")
        out.append(r"\begin{table}[htbp]")
        out.append(r"\centering")
        out.append(r"\begin{tabular}{" + "l" * len(header) + "}")
        out.append(r"\toprule")
        out.append(" & ".join(latex_escape(h) for h in header) + r" \\")
        out.append(r"\midrule")
        for row in tables[title]:
            out.append(" & ".join(latex_escape(c) for c in row) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{table}")
        out.append("")
    return "\n".join(out)


def run_validation() -> list[str]:
    results = []
    for label, path, column, query, expected in KNOWN_GOOD:
        if not path.exists():
            results.append(f"FAIL {label}: file missing ({path})")
            continue
        df = pd.read_csv(path)
        if query is not None:
            df = df.query(query)
        if df.empty or column not in df.columns:
            results.append(f"FAIL {label}: column/row not found")
            continue
        actual = float(df[column].iloc[0])
        ok = sig3(actual) == sig3(expected)
        results.append(
            f"{'PASS' if ok else 'FAIL'} {label}: got {actual:.6g}, expected {expected:.6g} (3 s.f.)"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", action="store_true", help="print LaTeX booktabs tables to stdout")
    args = parser.parse_args()

    abl = load_ablations()
    full = read_summary(FULL_INTERVAL_SUMMARY)
    cand = read_summary(CANDIDATE_SUMMARY)

    n_abl_rows = len(abl)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    snapshot_note = (
        f"Snapshot: {timestamp}; ablations.csv had {n_abl_rows} rows "
        f"({', '.join(sorted(abl['arm'].unique())) if n_abl_rows else 'no arms yet'}); "
        f"source: {ABLATIONS_CSV}"
    )

    tables = {
        TABLE_SPECS[0][0]: build_table1(abl, full, cand),
        TABLE_SPECS[1][0]: build_table2(abl, cand),
        TABLE_SPECS[2][0]: build_table3(abl),
    }
    validation = run_validation()

    if args.latex:
        print(render_latex(tables))
    else:
        OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_MD.write_text(render_markdown(tables, snapshot_note, validation), encoding="utf-8")
        print(f"wrote {OUTPUT_MD}")
        print(snapshot_note)
        for line in validation:
            print(line)


if __name__ == "__main__":
    main()
