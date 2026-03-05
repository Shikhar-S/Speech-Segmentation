from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

# -------------------- CONFIG --------------------

ROOT = Path("exp/runs/ipapack_ctc")
GLOBS = ["results-*.csv"]

DATASET_ORDER = [
    "gmuaccent",
    # "buckeye",
    # "epadb",
    # "voxangeles",
    "timit",
    # "speechoceannotth",
    "l2arctic_perceived",
]
KNOWN_DATASETS = set(DATASET_ORDER)

METHOD_MAP = [
    ("epitran", "epitran-g2p"),
    # ("xeus_multiaccent.accent_ls2.", "accent_mapping"),
    # ("xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze", "vanilla"),
    # ("xeus_multiaccent.panphon_ls2.", "panphon"),
    # ("xeus_multiaccent.schedule_panphonlsp2_4k_vanilla.", "panphon then vanilla"),
    # ("xeus_multiaccent.schedule_vanilla_4k_panphon", "vanilla then panphon"),
    # (
    #     "xeus_multiaccent.sched_vanilla_oracle_ls2",
    #     "vanilla then oracle",
    # ),
    ("xeus_multiaccent.interctc_l4_8_12.bs256", "fulldata_inter_CTC"),
    # (
    #     "xeus_multiaccent.losssched_half2.5k_m12to0.oracle.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40ksteps",
    #     "oracle_w_losssched",
    # ),
    (
        "xeus_multiaccent.losssched_half30k_m12tomp5.panphonk8.bs256.lr3em5.sched_p05warm_p75const_3kunfreeze.100ksteps",
        "fulldata_panphon_losssched",
    ),
    # (
    #     "xeus_huper.vanilla.bs128.lr3em5.sched_p10warm_p90const_500unfreeze.8ksteps",
    #     "huper-vanilla",
    # ),
    # (
    #     "xeus_multiaccent.oracle_ls7.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze.40kstep",
    #     "oracle_ls7",
    # ),
    # ("huper", "huper-model"),
    # ("koel", "koellabs-model"),
]

# All metrics to load.
METRIC_COLS = ["FER (%)", "PER (%)", "SUB (%)", "INS (%)", "DEL (%)"]

# Metrics that participate in bold/underline ranking (lower is better).
RANK_METRICS = {"FER (%)", "PER (%)"}

# Combined sub-line: which metrics, in order.
COMBINED_METRICS = ["PER (%)", "SUB (%)", "INS (%)", "DEL (%)"]
COMBINED_LABEL = "P|S|I|D"

EVAL_RE = re.compile(r"^(?P<method>.*?)-(?P<dataset>[^-]+?)(?:-(?P<ckpt>\d+))?$")
GT_STANDARD = "standard"
BASELINE_METHOD = "epitran-g2p"
EPS = 1e-9

# -------------------- PARSING --------------------


def parse_eval_name(s: str):
    m = EVAL_RE.match(str(s))
    if not m:
        return None
    return m.group("method"), m.group("dataset"), int(m.group("ckpt") or 0)


def match_method(method_raw: str) -> str | None:
    for needle, nice in METHOD_MAP:
        if needle in method_raw:
            return nice
    return None


def detect_variant(dataset: str) -> tuple[str, str]:
    """Map raw dataset name -> (base_dataset, gt_variant)."""
    if dataset in KNOWN_DATASETS:
        return dataset, GT_STANDARD
    for known in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if dataset.startswith(known + "_"):
            return known, dataset[len(known) + 1 :]
    return dataset, GT_STANDARD


# -------------------- LOADING --------------------


def load_results() -> pd.DataFrame:
    """Return long-form DataFrame:
    Method | ckpt | dataset | gt_variant | FER (%) | PER (%) | …
    """
    chunks: list[pd.DataFrame] = []

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

            df[["method_raw", "dataset_raw", "ckpt"]] = pd.DataFrame(
                parsed.dropna().tolist(), index=df.index
            )
            df["Method"] = df["method_raw"].map(match_method)
            df = df[df["Method"].notna()].copy()
            if df.empty:
                continue

            variants = df["dataset_raw"].map(detect_variant)
            df["dataset"] = variants.map(lambda x: x[0])
            df["gt_variant"] = variants.map(lambda x: x[1])
            df = df[df["dataset"].isin(KNOWN_DATASETS)].copy()

            df = df.sort_values(
                ["Method", "ckpt", "dataset", "gt_variant"]
            ).drop_duplicates(["Method", "ckpt", "dataset", "gt_variant"], keep="last")

            keep = ["Method", "ckpt", "dataset", "gt_variant"] + [
                c for c in METRIC_COLS if c in df.columns
            ]
            chunks.append(df[keep])

    if not chunks:
        raise SystemExit("No usable CSV rows found.")
    return pd.concat(chunks, ignore_index=True)


# -------------------- PIVOTING --------------------


def pivot_variant(df: pd.DataFrame, variant: str) -> dict[str, pd.DataFrame]:
    """Pivot long df -> {metric_name: wide_df} for one GT variant."""
    vdf = df[df["gt_variant"] == variant]
    pivots: dict[str, pd.DataFrame] = {}
    for metric in METRIC_COLS:
        if metric not in vdf.columns:
            continue
        wide = (
            vdf.pivot_table(
                index=["Method", "ckpt"],
                columns="dataset",
                values=metric,
                aggfunc="last",
            )
            .reindex(columns=DATASET_ORDER)
            .sort_index()
        )
        wide["avg"] = wide.mean(axis=1, skipna=True)
        pivots[metric] = wide
    return pivots


# -------------------- RANKINGS --------------------


def compute_rankings(
    pivots: dict[str, pd.DataFrame],
) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    """Return {(col, metric): (best, second)} excluding the baseline."""
    rankings: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for metric in RANK_METRICS:
        wide = pivots.get(metric)
        if wide is None:
            continue
        non_base = wide[~wide.index.get_level_values("Method").str.contains("epitran")]
        for col in wide.columns:
            vals = sorted(set(non_base[col].dropna().astype(float)))
            best = vals[0] if vals else None
            second = vals[1] if len(vals) > 1 else None
            rankings[(col, metric)] = (best, second)
    return rankings


# -------------------- FORMATTING --------------------


def _get_val(pivots: dict[str, pd.DataFrame], metric: str, idx, col):
    wide = pivots.get(metric)
    if wide is None or idx not in wide.index:
        return pd.NA
    return wide.loc[idx, col]


def _fmt_ranked(v, best, second, decimals: int = 2) -> str:
    if pd.isna(v):
        return ""
    v = float(v)
    s = f"{v:.{decimals}f}"
    if best is not None and abs(v - best) <= EPS:
        return f"[bold]{s}[/bold]"
    if second is not None and abs(v - second) <= EPS:
        return f"[u]{s}[/u]"
    return s


def _fmt_plain(v, decimals: int = 1) -> str:
    if pd.isna(v):
        return ""
    return f"{float(v):.{decimals}f}"


def fmt_cell(pivots, rankings, idx, ds, show_psid: bool = False) -> str:
    """Format a single table cell: FER on top, optionally P|S|I|D below."""
    # FER line
    fer_v = _get_val(pivots, "FER (%)", idx, ds)
    best, second = rankings.get((ds, "FER (%)"), (None, None))
    fer_str = _fmt_ranked(fer_v, best, second, decimals=2)

    if not show_psid:
        return fer_str

    # Combined line
    parts: list[str] = []
    any_present = False
    for metric in COMBINED_METRICS:
        v = _get_val(pivots, metric, idx, ds)
        if not pd.isna(v):
            any_present = True
        if metric in RANK_METRICS:
            b, s = rankings.get((ds, metric), (None, None))
            parts.append(_fmt_ranked(v, b, s, decimals=1))
        else:
            parts.append(_fmt_plain(v, decimals=1))

    if any_present:
        combined_str = "|".join(parts)
        return f"{fer_str}\n{combined_str}"
    elif fer_str:
        return fer_str
    else:
        return ""


# -------------------- RENDERING --------------------


def _add_rows(table, pivots, rankings, ds_cols, index_subset, show_psid: bool):
    for method, ckpt in index_subset:
        idx = (method, ckpt)
        cells: list[str] = [method, str(int(ckpt))]
        for ds in ds_cols:
            cells.append(fmt_cell(pivots, rankings, idx, ds, show_psid))
        table.add_row(*cells)


def _render_variant(table, pivots, rankings, ds_cols, show_psid: bool):
    fer_wide = pivots.get("FER (%)")
    if fer_wide is None or fer_wide.empty:
        return

    all_idx = fer_wide.index
    is_baseline = all_idx.get_level_values("Method") == BASELINE_METHOD

    non_base = all_idx[~is_baseline]
    if not non_base.empty:
        _add_rows(table, pivots, rankings, ds_cols, non_base, show_psid)

    base = all_idx[is_baseline]
    if not base.empty:
        table.add_section()
        _add_rows(table, pivots, rankings, ds_cols, base, show_psid)


def render_table(df: pd.DataFrame, show_psid: bool = False):
    console = Console()
    table = Table(title="Results by dataset", show_lines=False)

    table.add_column("Method")
    table.add_column("ckpt", justify="right")

    ds_cols = DATASET_ORDER + ["avg"]
    for ds in ds_cols:
        header = f"{ds}\n{COMBINED_LABEL}" if show_psid else ds
        table.add_column(header, justify="right")

    variants = sorted(df["gt_variant"].unique(), key=lambda v: (v != GT_STANDARD, v))

    for i, variant in enumerate(variants):
        pivots = pivot_variant(df, variant)
        rankings = compute_rankings(pivots)

        if i > 0:
            table.add_section()
            table.add_row(
                f"[italic]{variant} ground-truth[/italic]",
                "",
                *[""] * len(ds_cols),
            )

        _render_variant(table, pivots, rankings, ds_cols, show_psid)

    console.print(table)


# -------------------- CSV OUTPUT --------------------


def print_csv(df: pd.DataFrame):
    variants = sorted(df["gt_variant"].unique(), key=lambda v: (v != GT_STANDARD, v))
    for variant in variants:
        pivots = pivot_variant(df, variant)
        label = variant if variant != GT_STANDARD else "standard"
        for metric in METRIC_COLS:
            wide = pivots.get(metric)
            if wide is not None:
                print(f"\n--- CSV: {metric} ({label} ground-truth) ---")
                print(wide.reset_index().round(2).to_csv(index=False, na_rep=""))


# -------------------- MAIN --------------------


def main():
    parser = argparse.ArgumentParser(description="Display evaluation results.")
    parser.add_argument(
        "--psid",
        action="store_true",
        default=False,
        help="Show P|S|I|D (PER, SUB, INS, DEL) below FER in each cell",
    )
    args = parser.parse_args()

    df = load_results()
    mask = ~(df["Method"].str.contains("then")) & (df["ckpt"] == 4000)
    df = df[~mask]
    print_csv(df)
    render_table(df, show_psid=args.psid)


if __name__ == "__main__":
    main()
