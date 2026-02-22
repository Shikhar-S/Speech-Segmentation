from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = Path("exp/runs/ipapack_ctc")
GLOBS = ["results-*.csv"]

METRIC_COL = "PER (%)"
DATASET_ORDER = [
    "gmuaccent",
    "buckeye",
    "epadb",
    "speechoceannotth",
    "l2arctic_perceived",
]

# Keep only these methods (by substring match), map to readable names.
METHOD_MAP = [
    ("xeus_multiaccent.accent_ls2.", "accent_mapping"),
    ("xeus_multiaccent.bs256.lr3em5.sched_p15warm_p85const_3kunfreeze", "vanilla"),
    ("xeus_multiaccent.panphon_ls2.", "panphon"),
    ("xeus_multiaccent.schedule_panphonlsp2_4k_vanilla.", "panphon then vanilla"),
    ("xeus_multiaccent.schedule_vanilla_4k_panphon", "vanilla then panphon"),
    (
        "xeus_huper.vanilla.bs128.lr3em5.sched_p10warm_p90const_500unfreeze.8ksteps",
        "huper-vanilla",
    ),
    ("huper", "huper-model"),
]

EVAL_RE = re.compile(r"^(?P<method>.*?)-(?P<dataset>[^-]+?)(?:-(?P<ckpt>\d+))?$")


def parse_eval_name(s: str) -> tuple[str, str, int] | None:
    m = EVAL_RE.match(str(s))
    if not m:
        return None
    ckpt = int(m.group("ckpt")) if m.group("ckpt") is not None else 0
    return m.group("method"), m.group("dataset"), ckpt


def match_method(method_raw: str) -> str | None:
    for needle, nice in METHOD_MAP:
        if needle in method_raw:
            return nice
    return None


def main():
    rows = []
    for pat in GLOBS:
        for p in ROOT.glob(pat):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if (
                df.empty
                or "eval_name" not in df.columns
                or METRIC_COL not in df.columns
            ):
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

            # If duplicates exist for (Method, ckpt, dataset), keep last.
            df = df.sort_values(["Method", "ckpt", "dataset"]).drop_duplicates(
                subset=["Method", "ckpt", "dataset"],
                keep="last",
            )

            rows.append(df[["Method", "ckpt", "dataset", METRIC_COL]])

    if not rows:
        raise SystemExit(f"No usable CSV rows found under {ROOT} matching {GLOBS}")

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

    # ---- Rich table ----
    console = Console()
    table = Table(title=f"{METRIC_COL} by dataset (filtered methods)", show_lines=False)

    table.add_column("Method", overflow="fold")
    table.add_column("ckpt", justify="right")
    for ds in wide.columns:
        table.add_column(ds, justify="right")

    for (method, ckpt), r in wide.iterrows():
        vals = []
        for ds in wide.columns:
            v = r.get(ds)
            vals.append("" if pd.isna(v) else f"{float(v):.2f}")
        table.add_row(method, str(int(ckpt)), *vals)

    console.print(table)

    # ---- CSV to stdout ----
    # Make a flat DataFrame with Method/ckpt columns + dataset columns
    out = wide.reset_index()

    # Optional: round numeric columns nicely (keeps NaN)
    for c in DATASET_ORDER:
        if c in out.columns:
            out[c] = out[c].astype(float).round(2)

    print("\n--- CSV ---")
    # Prints to stdout as CSV (blank for NaNs)
    print(out.to_csv(index=False, na_rep=""))


if __name__ == "__main__":
    main()
