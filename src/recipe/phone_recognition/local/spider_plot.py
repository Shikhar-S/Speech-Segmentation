import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT_DIR = Path("/home/claude")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

FEAT_DISPLAY = {
    "syl": "Syllabic",
    "son": "Sonorant",
    "cons": "Consonantal",
    "cont": "Continuant",
    "delrel": "Del. Release",
    "lat": "Lateral",
    "nas": "Nasal",
    "strid": "Strident",
    "voi": "Voicing",
    "sg": "Spread Gl.",
    "cg": "Constr. Gl.",
    "ant": "Anterior",
    "cor": "Coronal",
    "distr": "Distributed",
    "lab": "Labial",
    "hi": "High",
    "lo": "Low",
    "back": "Back",
    "round": "Round",
    "tense": "Tense",
    "long": "Long",
}

FEATURE_TYPES = {
    "syl": "Major class",
    "son": "Major class",
    "cons": "Major class",
    "cont": "Manner of Articulation",
    "delrel": "Manner of Articulation",
    "lat": "Manner of Articulation",
    "nas": "Manner of Articulation",
    "strid": "Manner of Articulation",
    "voi": "Phonation",
    "sg": "Phonation",
    "cg": "Phonation",
    "ant": "Place of Articulation",
    "cor": "Place of Articulation",
    "distr": "Place of Articulation",
    "lab": "Place of Articulation",
    "hi": "Vowel",
    "lo": "Vowel",
    "back": "Vowel",
    "round": "Vowel",
    "tense": "Vowel",
    "long": "Quantity",
}

feat_df = pd.read_csv("/home/claude/feat_errors.csv")
a_name, b_name = "PhoneticXeus", "E-BranchFormer"
feat_a = (
    feat_df[feat_df.model == a_name]
    .set_index("feature")[["fer", "errors", "total"]]
    .to_dict("index")
)
feat_b = (
    feat_df[feat_df.model == b_name]
    .set_index("feature")[["fer", "errors", "total"]]
    .to_dict("index")
)

# Refined palette
COL_A = "#1D6FBF"  # deep blue
COL_B = "#C0392B"  # crimson
FILL_A = "#5BA3E0"
FILL_B = "#E57373"

# Subtitle badges per panel
PANEL_ICONS = {
    "Manner of Articulation": "",
    "Phonation": "",
    "Place of Articulation": "",
    "Vowel": "",
}
PANEL_BG = {
    "Manner of Articulation": "#EAF3FB",
    "Phonation": "#FEF3F2",
    "Place of Articulation": "#F0FBF0",
    "Vowel": "#FFF8EC",
}

TYPE_ORDER = ["Manner of Articulation", "Phonation", "Place of Articulation", "Vowel"]

fig = plt.figure(figsize=(14, 12), facecolor="white")
fig.patch.set_facecolor("white")

gs = GridSpec(2, 2, figure=fig, hspace=0.60, wspace=0.15)

for idx, ftype in enumerate(TYPE_ORDER):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs[row, col], polar=True)
    ax.set_facecolor(PANEL_BG[ftype])

    features = [
        f
        for f, t in FEATURE_TYPES.items()
        if t == ftype and f in feat_a and feat_a[f]["total"] > 0
    ]
    features = sorted(features, key=lambda f: -feat_a[f]["total"])

    labels = [FEAT_DISPLAY.get(f, f) for f in features]
    vals_a_grp = [feat_a[f]["fer"] * 100 for f in features]
    vals_b_grp = [feat_b[f]["fer"] * 100 for f in features]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    max_val = max(max(vals_a_grp), max(vals_b_grp))
    ylim = (int(max_val / 5) + 1) * 5
    ax.set_ylim(0, ylim)

    # ── Grid: polygon rings ───────────────────────────────────────────────
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines["polar"].set_visible(False)

    step = max(5, (ylim // 4 // 5) * 5)
    yticks = list(range(step, ylim + 1, step))

    # Shaded background polygon
    bg_angles = np.linspace(0, 2 * np.pi, 200)
    ax.fill(bg_angles, [ylim] * 200, color="white", alpha=0.55, zorder=0)

    # Polygon rings
    for ring_r in yticks:
        ring_angles = angles
        ax.plot(
            ring_angles,
            [ring_r] * len(ring_angles),
            color="#bbbbbb",
            linewidth=0.7,
            linestyle="--",
            zorder=1,
        )

    # Spoke lines
    for a in angles[:-1]:
        ax.plot([a, a], [0, ylim], color="#cccccc", linewidth=0.8, zorder=1)

    # ── Tick labels: manually placed exactly on each ring ────────────────
    ax.set_yticks([])  # disable default radial labels
    label_angle = np.deg2rad(180 / N + 8)  # place labels just past first spoke
    ax.text(
        label_angle,
        yticks[-1],
        f"{int(yticks[-1])}%",
        va="center",
        ha="left",
        fontsize=7.5,
        color="#555555",
        bbox=dict(boxstyle="square,pad=0", fc="white", alpha=0.85, ec="none"),
        zorder=6,
    )

    # ── Data polygons ─────────────────────────────────────────────────────
    for vals, col_line, col_fill, name in [
        (vals_a_grp, COL_A, FILL_A, a_name),
        (vals_b_grp, COL_B, FILL_B, b_name),
    ]:
        v = vals + vals[:1]
        ax.fill(angles, v, alpha=0.18, color=col_fill, zorder=2)
        ax.plot(angles, v, "-", linewidth=2.0, color=col_line, zorder=4, label=name)
        ax.scatter(
            angles[:-1],
            vals,
            s=50,
            color=col_line,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )

    # ── Axis labels ───────────────────────────────────────────────────────
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10.5, color="#222222")
    ax.tick_params(axis="x", pad=4)

    # ── Panel title ───────────────────────────────────────────────────────
    ax.set_title(
        f"{PANEL_ICONS[ftype]}{ftype}",
        fontsize=12.5,
        fontweight="bold",
        pad=26,
        color="#1a1a2e",
        loc="center",
    )

# ── Shared legend ─────────────────────────────────────────────────────────────
patch_a = mpatches.Patch(facecolor=FILL_A, edgecolor=COL_A, linewidth=1.5, label=a_name)
patch_b = mpatches.Patch(facecolor=FILL_B, edgecolor=COL_B, linewidth=1.5, label=b_name)

fig.legend(
    handles=[patch_a, patch_b],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=2,
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    fontsize=12,
    title="Model comparison  ·  Error Rate (%)",
    title_fontsize=11,
    handlelength=2.2,
    handleheight=1.2,
)

plt.savefig(OUT_DIR / "vox_spider_improved.pdf", bbox_inches="tight", facecolor="white")
print("Saved improved")
