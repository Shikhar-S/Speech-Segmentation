#!/usr/bin/env python3
"""Select on val/per_ctc, report test/per_ctc. Compact view (no duplicates).

Usage:
  python scripts/parse_wandb.py
  python scripts/parse_wandb.py --include '(arpa_vanilla|arpa_panphon|vanilla_interctc|joint_ctc_att|selfctc)'
  python scripts/parse_wandb.py --exclude 'debug|old'
  python scripts/parse_wandb.py --out_csv results.csv
"""
from __future__ import annotations

import argparse
import re
from typing import Optional

import pandas as pd
import wandb

# ── metric keys to try in order ──────────────────────────────────────────────
VAL_KEYS = ["val/per_ctc/dataloader_idx_0", "val/per_ctc"]
TEST_KEYS = ["test/per_ctc/dataloader_idx_1", "test/per_ctc"]
STEP_KEYS = ["trainer/global_step", "global_step", "_step"]

# ── method detection (first match wins; order matters) ────────────────────────
# NOTE: keep most specific first.
METHOD_RULES = [
    # interCTC (multi-layer explicitly encoded in name)
    (r"\bvanilla_interctc_l(?P<layers>[0-9_]+)\b", "ctc+interctc@multi"),
    # interCTC (single-layer / unspecified layer-set)
    (r"\bvanilla_interctc\b", "ctc+interctc@single"),
    # joint CTC+att
    (r"\bjoint_ctc_att\b", "joint_ctc_att"),
    # self-CTC (often also has l4_8_12)
    (r"\bselfctc_l(?P<layers>[0-9_]+)\b", "selfctc@multi"),
    (r"\bselfctc\b", "selfctc"),
    # panphon loss / arpa-panphon recipes
    (r"\barpa_panphon\b|(?<![a-z])panphon(?![a-z])", "ctc+panphon"),
    # baseline CTC recipes
    (r"\barpa_vanilla\b|(?<![a-z])vanilla(?![a-z])", "ctc"),
]


def parse_mix(name: str) -> Optional[float]:
    """Extract mixX.Y as percentage (0-100)."""
    m = re.search(r"mix([0-9]+(?:\.[0-9]+)?)", name, re.IGNORECASE)
    return float(m.group(1)) * 100 if m else None


def parse_layers(name: str) -> Optional[str]:
    """Extract '_l4_8_12' style layer-set as '4_8_12'."""
    m = re.search(r"_l([0-9_]+)\b", name, re.IGNORECASE)
    return m.group(1) if m else None


def parse_recipe_family(name_lower: str) -> str:
    """Optional: classify by folder-ish family for easier grouping."""
    if "timit_epitran_mix" in name_lower:
        return "timit_epitran_mix"
    if "timit_xeuspr_interctc" in name_lower:
        return "timit_xeuspr_interctc"
    if "timit_xeuspr_joint" in name_lower:
        return "timit_xeuspr_joint"
    if "timit_xeuspr_selfctc" in name_lower:
        return "timit_xeuspr_selfctc"
    return "other"


def parse_method(name: str) -> str:
    for pat, label in METHOD_RULES:
        if re.search(pat, name, re.IGNORECASE):
            return label
    return "other"


def select_on_val(run) -> Optional[dict]:
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

    run_name = run.name or ""
    run_name_lower = run_name.lower()

    return {
        # ---- unique identifiers (prevent duplicates) ----
        "run_id": run.id,
        "run_name": run_name,
        # ---- parsed grouping fields ----
        "family": parse_recipe_family(run_name_lower),
        "method": parse_method(run_name),
        "layers": parse_layers(run_name),
        "mix%": parse_mix(run_name),
        # ---- metric ----
        "test_per": test_per,
        # ---- useful metadata ----
        "state": getattr(run, "state", None),
        "created_at": str(getattr(run, "created_at", ""))[:19],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="Shikhar")
    ap.add_argument("--project", default="SSLPR")
    ap.add_argument("--must_contain", default="timit")
    ap.add_argument(
        "--include",
        default=r"(arpa_vanilla|arpa_panphon|vanilla_interctc|joint_ctc_att|selfctc)",
        help="Regex: include only run names matching this (case-insensitive).",
    )
    ap.add_argument(
        "--exclude",
        default=None,
        help="Regex: exclude run names matching this (case-insensitive).",
    )
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    inc = re.compile(args.include, re.IGNORECASE) if args.include else None
    exc = re.compile(args.exclude, re.IGNORECASE) if args.exclude else None

    api = wandb.Api()

    rows = []
    for r in api.runs(f"{args.entity}/{args.project}"):
        name = r.name or ""
        name_l = name.lower()

        if args.must_contain and args.must_contain.lower() not in name_l:
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
        .sort_values(["method", "layers", "mix%", "run_name"], na_position="last")
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
        table.add_column("layers")
        table.add_column("mix%", justify="right")
        table.add_column("test_per", justify="right", style="green")
        table.add_column("run_id", justify="right", style="dim")

        for _, row in df.iterrows():
            mix = f"{row['mix%']:.0f}" if pd.notna(row["mix%"]) else "—"
            per = f"{row['test_per']:.4f}" if pd.notna(row["test_per"]) else "—"
            layers = row["layers"] if pd.notna(row["layers"]) else "—"
            table.add_row(str(row["method"]), str(layers), mix, per, str(row["run_id"]))

        Console().print(table)
    except ImportError:
        # fallback
        cols = ["method", "layers", "mix%", "test_per", "run_id", "run_name"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
