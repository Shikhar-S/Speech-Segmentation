#!/usr/bin/env python3
"""Select on val/per_ctc, report test/per_ctc. Compact view.

Usage:
  python scripts/parse_wandb.py
  python scripts/parse_wandb.py --include '(vanilla|panphon|interctc)'
  python scripts/parse_wandb.py --out_csv results.csv
"""
import argparse
import re

import pandas as pd
import wandb

# ── metric keys to try in order ──────────────────────────────────────────────
VAL_KEYS = ["val/per_ctc/dataloader_idx_0", "val/per_ctc"]
TEST_KEYS = ["test/per_ctc/dataloader_idx_1", "test/per_ctc"]
STEP_KEYS = ["trainer/global_step", "global_step", "_step"]

# ── method detection (first match wins; order matters) ────────────────────────
METHOD_RULES = [
    (r"vanilla_interctc", "ctc+interctc"),
    (r"arpa_panphon|(?<![a-z])panphon", "panphon"),
    (r"arpa_vanilla|(?<![a-z])vanilla", "ctc"),
]


def parse_mix(name: str):
    m = re.search(r"mix([0-9]+(?:\.[0-9]+)?)", name)
    return float(m.group(1)) * 100 if m else None


def parse_method(name: str) -> str:
    for pat, label in METHOD_RULES:
        if re.search(pat, name, re.IGNORECASE):
            return label
    return "other"


def select_on_val(run) -> dict | None:
    """Best step = argmin(val/per_ctc); report test/per_ctc at that step."""
    df = run.history()
    if df is None or df.empty:
        return None

    val_key = next((k for k in VAL_KEYS if k in df.columns), None)
    test_key = next((k for k in TEST_KEYS if k in df.columns), None)
    step_key = next((k for k in STEP_KEYS if k in df.columns), None)
    if step_key is None:
        df = df.copy()
        df["_row"] = range(len(df))
        step_key = "_row"
    if val_key is None:
        return None

    df_val = df.dropna(subset=[val_key])
    if df_val.empty:
        return None

    best = df_val.loc[df_val[val_key].idxmin()]
    best_step = int(best[step_key]) if pd.notna(best[step_key]) else None

    test_per = None
    if test_key and best_step is not None:
        hits = df.loc[df[step_key] == best_step, test_key].dropna()
        if not hits.empty:
            test_per = float(hits.iloc[0])

    return {
        "method": parse_method(run.name),
        "mix%": parse_mix(run.name),
        "test_per": test_per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="Shikhar")
    ap.add_argument("--project", default="SSLPR")
    ap.add_argument("--must_contain", default="timit")
    ap.add_argument(
        "--include", default=r"(arpa_vanilla|arpa_panphon|vanilla_interctc)"
    )
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    inc = re.compile(args.include, re.IGNORECASE) if args.include else None
    exc = re.compile(args.exclude, re.IGNORECASE) if args.exclude else None

    rows = []
    for r in wandb.Api().runs(f"{args.entity}/{args.project}"):
        name = (r.name or "").lower()
        if args.must_contain and args.must_contain not in name:
            continue
        if inc and not inc.search(name):
            continue
        if exc and exc.search(name):
            continue
        try:
            row = select_on_val(r)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"[WARN] {r.name}: {e}")

    if not rows:
        print("No matching runs found.")
        return

    df = (
        pd.DataFrame(rows)
        .sort_values(["method", "mix%"], na_position="last")
        .reset_index(drop=True)
    )

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"Saved → {args.out_csv}")

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(
            title=f"{args.entity}/{args.project}",
            show_lines=False,
            header_style="bold cyan",
        )
        table.add_column("method", style="bold")
        table.add_column("mix%", justify="right")
        table.add_column("test_per", justify="right", style="green")

        for _, row in df.iterrows():
            mix = f"{row['mix%']:.0f}" if pd.notna(row["mix%"]) else "—"
            per = f"{row['test_per']:.4f}" if pd.notna(row["test_per"]) else "—"
            table.add_row(str(row["method"]), mix, per)

        Console().print(table)
    except ImportError:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
