"""GMU Accent Archive — FER scatter: PhoneticXeus vs. E-BranchFormer.

Reads the per-utterance cache CSV, aggregates FER by accent and model,
and produces a scatter plot saved to is26_viz/gmu_fer_scatter.pdf.
Also prints a LaTeX table and \\newcommand block to stdout.

Usage::

    MPLBACKEND=Agg python src/recipe/phone_recognition/local/gmuaccent_fer_plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_HERE = Path(__file__).parent
_CSV_IN = _HERE / "is26_viz" / "cache_gmu_per_utt.csv"
OUT_DIR = _HERE / "is26_viz"

# ── colours ──────────────────────────────────────────────────────────────────
_BLUE = "#2c7bb6"   # PhoneticXeus wins
_RED  = "#d7191c"   # E-BranchFormer wins
_GREY = "#888888"   # identity line


def _load_agg(csv_path: Path) -> pd.DataFrame:
    """Return accent-level mean FER pivoted wide, with a delta column."""
    df = pd.read_csv(csv_path)
    agg = (
        df.groupby(["langcode", "model"])["fer"]
        .mean()
        .reset_index()
        .pivot(index="langcode", columns="model", values="fer")
        .rename_axis(None, axis=1)
        .reset_index()
    )
    agg["delta"] = agg["E-BranchFormer"] - agg["PhoneticXeus"]   # +ve → PX wins
    agg["px_wins"] = agg["delta"] > 0
    return agg


def _make_scatter(agg: pd.DataFrame) -> None:
    """Save the scatter plot to OUT_DIR/gmu_fer_scatter.pdf."""
    px_n  = agg["px_wins"].sum()
    eb_n  = (~agg["px_wins"]).sum()

    vals = agg[["PhoneticXeus", "E-BranchFormer"]]
    lim_lo = vals.min().min() * 0.95
    lim_hi = vals.max().max() * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))

    # Shaded region below diagonal (PhoneticXeus better)
    ax.fill_between(
        [lim_lo, lim_hi], [lim_lo, lim_hi], lim_lo,
        color=_BLUE, alpha=0.05, zorder=0,
    )

    # Identity line
    ax.plot(
        [lim_lo, lim_hi], [lim_lo, lim_hi],
        "--", color=_GREY, linewidth=1.2, label="Equal performance", zorder=1,
    )

    # Scatter: PX wins (blue) first, then EB wins (red) on top
    for flag, color, label in [
        (True,  _BLUE, f"PhoneticXeus better (N={px_n})"),
        (False, _RED,  f"E-BranchFormer better (N={eb_n})"),
    ]:
        sub = agg[agg["px_wins"] == flag]
        ax.scatter(
            sub["E-BranchFormer"], sub["PhoneticXeus"],
            color=color, s=30, alpha=0.75, linewidths=0,
            label=label, zorder=2,
        )

    # Annotate the 5 EB-wins accents
    eb_wins = agg[~agg["px_wins"]].copy()
    for _, row in eb_wins.iterrows():
        ax.annotate(
            row["langcode"],
            xy=(row["E-BranchFormer"], row["PhoneticXeus"]),
            fontsize=7.5, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
        )

    # Annotate top-3 PX-wins (largest delta)
    top3 = agg[agg["px_wins"]].nlargest(3, "delta")
    for _, row in top3.iterrows():
        ax.annotate(
            row["langcode"],
            xy=(row["E-BranchFormer"], row["PhoneticXeus"]),
            fontsize=7.5, ha="left", va="top",
            xytext=(4, -4), textcoords="offset points",
            color=_BLUE,
        )

    # Region label
    ax.text(
        lim_hi * 0.97, lim_lo * 1.15,
        "\u2190 PhoneticXeus better",
        fontsize=8, color=_BLUE, ha="right", style="italic",
    )

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("E-BranchFormer FER (%)", labelpad=8)
    ax.set_ylabel("PhoneticXeus FER (%)", labelpad=8)
    ax.legend(frameon=True, framealpha=0.9, fontsize=9, loc="upper left")
    sns.despine(ax=ax)

    plt.tight_layout()
    out = OUT_DIR / "gmu_fer_scatter.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close(fig)


def _print_latex(agg: pd.DataFrame) -> None:
    """Print a LaTeX table and \\newcommand block to stdout."""
    n_accents = len(agg)
    px_mean   = agg["PhoneticXeus"].mean()
    eb_mean   = agg["E-BranchFormer"].mean()
    px_wins_n = int(agg["px_wins"].sum())
    eb_wins_n = n_accents - px_wins_n
    mean_delta = agg["delta"].mean()

    # Largest PX gain
    best_px = agg[agg["px_wins"]].loc[agg[agg["px_wins"]]["delta"].idxmax()]
    best_px_accent = best_px["langcode"].capitalize()
    best_px_delta  = best_px["delta"]

    # Largest EB gain (most negative delta)
    best_eb = agg[~agg["px_wins"]].loc[agg[~agg["px_wins"]]["delta"].idxmin()]
    best_eb_accent = best_eb["langcode"].capitalize()
    best_eb_delta  = abs(best_eb["delta"])

    table = rf"""
\begin{{table}}[t]
\centering
\caption{{Accent-level mean FER (\%) on the GMU Accent Archive (192 accents).
  $\Delta$ = E-BranchFormer $-$ PhoneticXeus (positive = PhoneticXeus wins).}}
\label{{tab:gmu_fer}}
\begin{{tabular}}{{lccc}}
\toprule
Model & Mean FER (\%) & Accents improved & Largest gain \\
\midrule
PhoneticXeus   & {px_mean:.2f} & {px_wins_n} / {n_accents} & +{best_px_delta:.1f} pp ({best_px_accent}) \\
E-BranchFormer & {eb_mean:.2f} & {eb_wins_n} / {n_accents} & +{best_eb_delta:.1f} pp ({best_eb_accent}) \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    commands = rf"""
% --- GMU Accent Archive stats ---
\newcommand{{\GmuPxFer}}{{{px_mean:.2f}}}
\newcommand{{\GmuEbFer}}{{{eb_mean:.2f}}}
\newcommand{{\GmuPxWins}}{{{px_wins_n}}}          % out of {n_accents}
\newcommand{{\GmuMeanDelta}}{{{mean_delta:.2f}}}   % mean EB - PX gap in pp
\newcommand{{\GmuMaxDelta}}{{{best_px_delta:.2f}}} % largest PX gain
\newcommand{{\GmuMaxDeltaAccent}}{{{best_px_accent}}}
"""
    print(table)
    print(commands)


def main() -> None:
    """Entry point."""
    agg = _load_agg(_CSV_IN)
    _make_scatter(agg)
    _print_latex(agg)


if __name__ == "__main__":
    main()
