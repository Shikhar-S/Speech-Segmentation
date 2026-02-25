#!/usr/bin/env python3
"""
Select-on-val, report-on-test for W&B runs.

What it does:
- Filter runs by name substring (default: "timit")
- Include/exclude runs by regex (so you can drop "fix" without hardcoding)
- Extract:
  - mix% from "mixX" token (mix0.4 => 40%)
  - method by matching regex rules (configurable)
- For each run:
  - best_step = argmin(val/per_ctc)
  - report test/per_ctc at that same step
- Output:
  1) CSV printed to stdout (always)
  2) Rich console table (optional, default on)
  3) Optionally save to CSV file

Usage:
  python report_timit_table.py --entity Shikhar --project SSLPR

Exclude runs containing "fix":
  python report_timit_table.py --exclude '(^|[._-])fix([._-]|$)'

Only include runs that match "arpa_(vanilla|panphon)":
  python report_timit_table.py --include 'arpa_(vanilla|panphon)'

Dump to file:
  python report_timit_table.py --out_csv results.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, List, Tuple, Pattern

import pandas as pd
import wandb


# ---------------------------
# Data model
# ---------------------------


@dataclass
class RunResult:
    run_id: str
    run_name: str
    method: str
    mix_percent: Optional[float]
    best_step: Optional[int]
    best_val_per_ctc: Optional[float]
    test_per_ctc_at_best_val: Optional[float]
    val_key_used: Optional[str]
    test_key_used: Optional[str]


# ---------------------------
# Parsing helpers
# ---------------------------


def parse_mix_percent(run_name: str) -> Optional[float]:
    """Extract x from 'mixx' in the run name; interpret as x*100 percent."""
    m = re.search(r"(?:^|[._-])mix([0-9]+(?:\.[0-9]+)?)", run_name)
    if not m:
        return None
    return float(m.group(1)) * 100.0


def build_method_parser(rules: List[Tuple[Pattern[str], str]]):
    """
    Returns a function that maps run_name -> method label by applying first matching regex rule.
    """

    def _parse(run_name: str) -> str:
        name = run_name.lower()
        for pat, label in rules:
            if pat.search(name):
                return label
        return "unknown"

    return _parse


# ---------------------------
# Filtering helpers
# ---------------------------


def compile_optional_regex(
    s: Optional[str], flags=re.IGNORECASE
) -> Optional[Pattern[str]]:
    if not s:
        return None
    return re.compile(s, flags=flags)


def run_name_passes_filters(
    name: str,
    must_contain: Optional[str],
    include_re: Optional[Pattern[str]],
    exclude_re: Optional[Pattern[str]],
) -> bool:
    lname = (name or "").lower()
    if must_contain and must_contain.lower() not in lname:
        return False
    if include_re and not include_re.search(lname):
        return False
    if exclude_re and exclude_re.search(lname):
        return False
    return True


# ---------------------------
# W&B helpers
# ---------------------------


def fetch_runs(entity: str, project: str) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    return list(api.runs(f"{entity}/{project}"))


def choose_metric_keys(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick best matching val/test keys from available history columns.
    Preference order:
      val:  val/per_ctc/dataloader_idx_0, val/per_ctc
      test: test/per_ctc/dataloader_idx_1, test/per_ctc
    """
    cols = set(map(str, df.columns))

    val_candidates = [
        "val/per_ctc/dataloader_idx_0",
        "val/per_ctc",
    ]
    test_candidates = [
        "test/per_ctc/dataloader_idx_1",
        "test/per_ctc",
    ]

    val_key = next((k for k in val_candidates if k in cols), None)
    test_key = next((k for k in test_candidates if k in cols), None)
    return val_key, test_key


def select_on_val_report_test(
    run: wandb.apis.public.Run,
    parse_method,
    step_candidates: Sequence[str] = ("trainer/global_step", "global_step", "_step"),
) -> RunResult:
    df = run.history()
    if df is None or df.empty:
        return RunResult(
            run_id=run.id,
            run_name=run.name,
            method=parse_method(run.name),
            mix_percent=parse_mix_percent(run.name),
            best_step=None,
            best_val_per_ctc=None,
            test_per_ctc_at_best_val=None,
            val_key_used=None,
            test_key_used=None,
        )

    val_key, test_key = choose_metric_keys(df)
    step_key = next((k for k in step_candidates if k in df.columns), None)

    if step_key is None:
        step_key = "_row"
        df = df.copy()
        df[step_key] = range(len(df))

    if val_key is None:
        return RunResult(
            run_id=run.id,
            run_name=run.name,
            method=parse_method(run.name),
            mix_percent=parse_mix_percent(run.name),
            best_step=None,
            best_val_per_ctc=None,
            test_per_ctc_at_best_val=None,
            val_key_used=None,
            test_key_used=test_key,
        )

    df_val = df.dropna(subset=[val_key])
    if df_val.empty:
        return RunResult(
            run_id=run.id,
            run_name=run.name,
            method=parse_method(run.name),
            mix_percent=parse_mix_percent(run.name),
            best_step=None,
            best_val_per_ctc=None,
            test_per_ctc_at_best_val=None,
            val_key_used=val_key,
            test_key_used=test_key,
        )

    best_idx = df_val[val_key].idxmin()
    best_row = df_val.loc[best_idx]
    best_step = int(best_row[step_key]) if pd.notna(best_row[step_key]) else None
    best_val = float(best_row[val_key])

    best_test = None
    if test_key is not None and best_step is not None:
        same_step = df.loc[df[step_key] == best_step, test_key].dropna()
        if not same_step.empty:
            best_test = float(same_step.iloc[0])

    return RunResult(
        run_id=run.id,
        run_name=run.name,
        method=parse_method(run.name),
        mix_percent=parse_mix_percent(run.name),
        best_step=best_step,
        best_val_per_ctc=best_val,
        test_per_ctc_at_best_val=best_test,
        val_key_used=val_key,
        test_key_used=test_key,
    )


# ---------------------------
# Reporting
# ---------------------------


def results_to_df(results: List[RunResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])

    # order + nice display columns
    display_cols = [
        "run_name",
        "method",
        "mix_percent",
        "best_step",
        "best_val_per_ctc",
        "test_per_ctc_at_best_val",
        "val_key_used",
        "test_key_used",
        "run_id",
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    # sort to compare methods/mix easily
    sort_cols = [
        c for c in ["method", "mix_percent", "best_val_per_ctc"] if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")

    return df


def print_csv_to_stdout(df: pd.DataFrame) -> None:
    # Always print CSV first (as you requested)
    print(df.to_csv(index=False).rstrip("\n"))


def print_rich_table(
    df: pd.DataFrame, title: str = "W&B val-selected test PER"
) -> None:
    # Rich is optional dependency; keep script usable without it
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print(
            "\n[rich not installed] Falling back to plain table. Install with: pip install rich\n"
        )
        pd.set_option("display.max_colwidth", 140)
        pd.set_option("display.width", 240)
        pd.set_option("display.max_rows", None)
        print(df.to_string(index=False))
        return

    console = Console()
    table = Table(title=title, show_lines=False, header_style="bold")

    # Add columns
    for col in df.columns:
        justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
        table.add_column(
            col, justify=justify, overflow="fold", no_wrap=(col == "run_id")
        )

    # Add rows
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            v = row[col]
            if pd.isna(v):
                cells.append(Text("NA", style="dim"))
            elif isinstance(v, float):
                # compact numeric formatting
                cells.append(f"{v:.6g}")
            else:
                cells.append(str(v))
        table.add_row(*cells)

    console.print()
    console.print(table)
    console.print()


# ---------------------------
# Main
# ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="Shikhar")
    ap.add_argument("--project", default="SSLPR")

    # selection controls (so you can re-use script for other comparisons)
    ap.add_argument(
        "--must_contain",
        default="timit",
        help="Run name must contain this substring (case-insensitive).",
    )
    ap.add_argument(
        "--include",
        default=r"arpa_(vanilla|panphon)",
        help="Regex: run name must match (applied after must_contain).",
    )
    ap.add_argument(
        "--exclude",
        default=r"(^|[._-])fix([._-]|$)",
        help="Regex: run name must NOT match. Default excludes 'fix' runs.",
    )

    # output controls
    ap.add_argument(
        "--out_csv", default=None, help="If set, also write results to this CSV path."
    )
    ap.add_argument("--no_rich", action="store_true", help="Disable rich output table.")
    args = ap.parse_args()

    include_re = compile_optional_regex(args.include)
    exclude_re = compile_optional_regex(args.exclude)

    # method rules are configurable (not hardcoded to a single project layout)
    method_rules = [
        (
            re.compile(r"(?:^|[._-])arpa_vanilla(?:[._-]|$)", re.IGNORECASE),
            "ctc (vanilla)",
        ),
        (
            re.compile(r"(?:^|[._-])arpa_panphon(?:[._-]|$)", re.IGNORECASE),
            "panphon (proposed)",
        ),
        (re.compile(r"(?:^|[._-])vanilla(?:[._-]|$)", re.IGNORECASE), "ctc (vanilla)"),
        (
            re.compile(r"(?:^|[._-])panphon(?:[._-]|$)", re.IGNORECASE),
            "panphon (proposed)",
        ),
    ]
    parse_method = build_method_parser(method_rules)

    runs = fetch_runs(args.entity, args.project)

    picked = [
        r
        for r in runs
        if run_name_passes_filters(
            r.name or "", args.must_contain, include_re, exclude_re
        )
    ]

    results: List[RunResult] = []
    for run in picked:
        try:
            results.append(select_on_val_report_test(run, parse_method=parse_method))
        except Exception as e:
            # keep going
            results.append(
                RunResult(
                    run_id=run.id,
                    run_name=run.name,
                    method=parse_method(run.name),
                    mix_percent=parse_mix_percent(run.name),
                    best_step=None,
                    best_val_per_ctc=None,
                    test_per_ctc_at_best_val=None,
                    val_key_used=None,
                    test_key_used=None,
                )
            )
            print(f"[WARN] Failed on run '{run.name}': {e}")

    df = results_to_df(results)

    # 1) print as CSV to console
    print_csv_to_stdout(df)

    # optionally write csv file
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)

    # 2) rich console output
    if not args.no_rich:
        print_rich_table(
            df,
            title=f"{args.entity}/{args.project} | must_contain='{args.must_contain}'",
        )


if __name__ == "__main__":
    main()
