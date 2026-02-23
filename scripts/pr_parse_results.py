from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

# -------------------- CONFIG --------------------

ROOT = Path("exp/runs/ipapack_ctc")
GLOBS = ["results-*.csv"]

METRIC_COL = "FER (%)"

DATASET_ORDER = [
    "gmuaccent",
    "buckeye",
    "epadb",
    "speechoceannotth",
    "l2arctic_perceived",
]

METHOD_MAP = [
    ("epitran", "epitran-g2p"),
    ("xeus_multiaccent.accent_ls2.", "accent_mapping"),
    ("xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze", "vanilla"),
    ("xeus_multiaccent.panphon_ls2.", "panphon"),
    ("xeus_multiaccent.schedule_panphonlsp2_4k_vanilla.", "panphon then vanilla"),
    ("xeus_multiaccent.schedule_vanilla_4k_panphon", "vanilla then panphon"),
    (
        "xeus_huper.vanilla.bs128.lr3em5.sched_p10warm_p90const_500unfreeze.8ksteps",
        "huper-vanilla",
    ),
    (
        "xeus_multiaccent.oracle_ls7.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40kstep",
        "oracle_ls7",
    ),
    ("huper", "huper-model"),
    ("koel", "koellabs-model"),
]

EVAL_RE = re.compile(r"^(?P<method>.*?)-(?P<dataset>[^-]+?)(?:-(?P<ckpt>\d+))?$")


# -------------------- PARSING --------------------


def parse_eval_name(s: str):
    m = EVAL_RE.match(str(s))
    if not m:
        return None
    ckpt = int(m.group("ckpt") or 0)
    return m.group("method"), m.group("dataset"), ckpt


def match_method(method_raw: str):
    for needle, nice in METHOD_MAP:
        if needle in method_raw:
            return nice
    return None


# -------------------- LOADING --------------------


def load_results() -> pd.DataFrame:
    rows = []
    for pat in GLOBS:
        for p in ROOT.glob(pat):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty or "eval_name" not in df.columns:
                continue

            parsed = df["eval_name"].map(parse_eval_name)
            df = df[parsed.notna()].copy()
            if df.empty:
                continue

            df[["method_raw", "dataset", "ckpt"]] = pd.DataFrame(
                parsed.dropna().tolist(), index=df.index
            )
            df["Method"] = df["method_raw"].map(match_method)
            df = df[df["Method"].notna()].copy()
            if df.empty:
                continue

            df = df.sort_values(["Method", "ckpt", "dataset"]).drop_duplicates(
                ["Method", "ckpt", "dataset"], keep="last"
            )

            rows.append(df[["Method", "ckpt", "dataset", METRIC_COL]])

    if not rows:
        raise SystemExit("No usable CSV rows found.")

    all_df = pd.concat(rows, ignore_index=True)

    wide = (
        all_df.pivot_table(
            index=["Method", "ckpt"],
            columns="dataset",
            values=METRIC_COL,
            aggfunc="last",
        )
        .reindex(columns=DATASET_ORDER)
        .sort_index()
    )

    wide["avg"] = wide.mean(axis=1, skipna=True)
    return wide


# -------------------- BEST / SECOND --------------------


def compute_best_second(wide: pd.DataFrame):
    """Compute best/second-best excluding epitran."""
    non_epi = wide[~wide.index.get_level_values("Method").str.contains("epitran")]

    best_second = {}
    for c in wide.columns:
        vals = non_epi[c].dropna().astype(float)
        if len(vals) == 0:
            best_second[c] = (None, None)
            continue
        uniq = sorted(set(vals))
        best = uniq[0]
        second = uniq[1] if len(uniq) > 1 else None
        best_second[c] = (best, second)
    return best_second


# -------------------- PRINTING --------------------


def render_table(wide: pd.DataFrame):
    console = Console()
    table = Table(title=f"{METRIC_COL} by dataset", show_lines=False)

    table.add_column("Method")
    table.add_column("ckpt", justify="right")
    for c in wide.columns:
        table.add_column(c, justify="right")

    best_second = compute_best_second(wide)
    EPS = 1e-9

    def fmt(v, c):
        if pd.isna(v):
            return ""
        v = float(v)
        s = f"{v:.2f}"
        best, second = best_second[c]
        if best is not None and abs(v - best) <= EPS:
            return f"[bold]{s}[/bold]"
        if second is not None and abs(v - second) <= EPS:
            return f"[u]{s}[/u]"
        return s

    # Split epitran and others
    epi_rows = wide[wide.index.get_level_values("Method") == "epitran-g2p"]
    other_rows = wide[wide.index.get_level_values("Method") != "epitran-g2p"]

    for (m, ckpt), r in other_rows.iterrows():
        table.add_row(m, str(int(ckpt)), *[fmt(r[c], c) for c in wide.columns])

    if not epi_rows.empty:
        table.add_section()
        for (m, ckpt), r in epi_rows.iterrows():
            table.add_row(m, str(int(ckpt)), *[fmt(r[c], c) for c in wide.columns])

    console.print(table)


# -------------------- MAIN --------------------


def main():
    wide = load_results()
    render_table(wide)

    print("\n--- CSV ---")
    print(wide.reset_index().round(2).to_csv(index=False, na_rep=""))


if __name__ == "__main__":
    main()
