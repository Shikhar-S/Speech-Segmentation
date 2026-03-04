"""Standalone script: language-family PFER bar chart for VoxAngeles.

Reads from the cached CSV if present, otherwise builds from per-utterance data.
Output: is26_viz/vox_family.pdf
"""

import sys

sys.path.insert(0, "/work/nvme/bbjs/sbharadwaj/powsm/xeuspr")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from src.recipe.phone_recognition.local.error_analysis_utils import (
    annotate_lang_meta,
    load_voxangeles_lang_meta,
)

OUT_DIR = Path("src/recipe/phone_recognition/local/is26_viz")

a_name = "PhoneticXeus"
b_name = "E-BranchFormer"
PALETTE = {a_name: "#2c7bb6", b_name: "#d7191c"}

_CSV_FAMILY = OUT_DIR / "cache_vox_family_fer.csv"
_CSV_PER_UTT = OUT_DIR / "cache_vox_per_utt.csv"

# Load family-level FER; rebuild from per-utt cache if needed
if _CSV_FAMILY.exists():
    plot_df = pd.read_csv(_CSV_FAMILY)
else:
    df_all = pd.read_csv(_CSV_PER_UTT)
    lang_meta = load_voxangeles_lang_meta()
    df_all = annotate_lang_meta(df_all, lang_meta)
    # Basque (eus) is a language isolate — no family in Glottolog; label explicitly
    df_all["family"] = df_all["family"].fillna("Basque")

    # Mean FER per (model, family), then sort by MODEL_A FER ascending
    fer_comparison = (
        df_all.groupby(["model", "family"])["fer"]
        .mean()
        .reset_index()
        .pivot(index="family", columns="model", values="fer")
        .reset_index()
    )
    plot_df = fer_comparison.sort_values(a_name).melt(
        id_vars="family",
        value_vars=[a_name, b_name],
        var_name="Model",
        value_name="FER (%)",
    )
    plot_df.to_csv(_CSV_FAMILY, index=False)
    print(f"Cached → {_CSV_FAMILY}")

# Sort families by MODEL_A FER for consistent ordering
family_order = (
    plot_df[plot_df["Model"] == a_name].sort_values("FER (%)")["family"].tolist()
)
plot_df["family"] = pd.Categorical(
    plot_df["family"], categories=family_order, ordered=True
)
plot_df = plot_df.sort_values("family")

fig, ax = plt.subplots(figsize=(16, 5.5))

sns.barplot(
    data=plot_df,
    x="family",
    y="FER (%)",
    hue="Model",
    palette=PALETTE,
    width=0.65,
    ax=ax,
)

# Value labels on each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f%%", fontsize=8, padding=2, color="#333333")

ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, linestyle="-")
ax.set_axisbelow(True)
ax.set_xlabel("")
ax.set_ylabel("PFER (%)", labelpad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9.5)
ax.legend(title="Model", frameon=True, loc="upper left", framealpha=0.9)
ax.margins(x=0.02)
sns.despine(bottom=True)

plt.tight_layout()
plt.savefig(OUT_DIR / "vox_family.pdf", bbox_inches="tight")
print(f"Saved → {OUT_DIR / 'vox_family.pdf'}")
