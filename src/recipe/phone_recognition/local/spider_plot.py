import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT_DIR = Path("src/recipe/phone_recognition/local/is26_viz")

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})

# Short display names for panphon features
FEAT_DISPLAY = {
    "syl": "Syllabic",    "son": "Sonorant",    "cons": "Consonantal",
    "cont": "Continuant", "delrel": "Delayed Release", "lat": "Lateral",
    "nas": "Nasal",       "strid": "Strident",
    "voi": "Voicing",     "sg": "Spread Glottis",  "cg": "Constrained Glottis",
    "ant": "Anterior",    "cor": "Coronal",    "distr": "Distributed", "lab": "Labial",
    "hi": "High",  "lo": "Low",  "back": "Back",  "round": "Round",  "tense": "Tense",
    "long": "Long",
}

# Group each feature into one of the four spider panels
FEATURE_TYPES = {
    "syl": "Major class",  "son": "Major class",  "cons": "Major class",
    "cont": "Manner of Articulation", "delrel": "Manner of Articulation",
    "lat":  "Manner of Articulation", "nas":    "Manner of Articulation",
    "strid":"Manner of Articulation",
    "voi": "Phonation", "sg": "Phonation", "cg": "Phonation",
    "ant": "Place of Articulation", "cor":   "Place of Articulation",
    "distr":"Place of Articulation", "lab":  "Place of Articulation",
    "hi": "Vowel", "lo": "Vowel", "back": "Vowel", "round": "Vowel", "tense": "Vowel",
    "long": "Quantity",
}

feat_df = pd.read_csv(OUT_DIR / "cache_vox_feat_errors.csv")
a_name, b_name = "PhoneticXeus", "E-BranchFormer"

# Index FER/count data by feature name for fast lookup
feat_a = feat_df[feat_df.model == a_name].set_index("feature")[["fer", "errors", "total"]].to_dict("index")
feat_b = feat_df[feat_df.model == b_name].set_index("feature")[["fer", "errors", "total"]].to_dict("index")

COL_A, FILL_A = "#1D6FBF", "#5BA3E0"   # deep blue  (PhoneticXeus)
COL_B, FILL_B = "#C0392B", "#E57373"   # crimson    (E-BranchFormer)

PANEL_BG = {
    "Manner of Articulation": "#EAF3FB",
    "Phonation":              "#FEF3F2",
    "Place of Articulation":  "#F0FBF0",
    "Vowel":                  "#FFF8EC",
}
TYPE_ORDER = ["Manner of Articulation", "Phonation", "Place of Articulation", "Vowel"]

fig = plt.figure(figsize=(10, 10), facecolor="white")
gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.01,
               top=0.93, bottom=0.08, left=0.04, right=0.96)

for idx, ftype in enumerate(TYPE_ORDER):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs[row, col], polar=True)
    ax.set_facecolor(PANEL_BG[ftype])

    # Features for this panel, sorted descending by observation count
    features = sorted(
        [f for f, t in FEATURE_TYPES.items() if t == ftype and f in feat_a and feat_a[f]["total"] > 0],
        key=lambda f: -feat_a[f]["total"],
    )
    labels      = [FEAT_DISPLAY.get(f, f) for f in features]
    vals_a_grp  = [feat_a[f]["fer"] * 100 for f in features]
    vals_b_grp  = [feat_b[f]["fer"] * 100 for f in features]

    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]   # closed

    # Round ylim up to next multiple of 5
    ylim = (int(max(max(vals_a_grp), max(vals_b_grp)) / 5) + 1) * 5
    ax.set_ylim(-ylim * 0.15, ylim)   # negative lower bound shrinks dead-centre space

    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines["polar"].set_visible(False)

    step   = max(5, (ylim // 4 // 5) * 5)
    yticks = list(range(step, ylim + 1, step))

    # White fill behind data so panel bg doesn't bleed through
    ax.fill(np.linspace(0, 2 * np.pi, 200), [ylim] * 200, color="white", alpha=0.55, zorder=0)

    # Dashed concentric polygon rings
    for ring_r in yticks:
        ax.plot(angles, [ring_r] * len(angles), color="#888888", linewidth=1.0, linestyle="--", zorder=1)

    # Radial spokes
    for a in angles[:-1]:
        ax.plot([a, a], [0, ylim], color="#cccccc", linewidth=0.8, zorder=1)

    # Ring labels on the spoke nearest to vertical (θ ≈ π/2)
    ax.set_yticks([])
    label_spoke_idx = min(range(N), key=lambda i: abs(angles[i] - np.pi / 2))
    label_angle = angles[label_spoke_idx]
    label_ha    = "right" if label_angle <= np.pi / 2 else "left"
    for r in yticks:
        t = ax.text(label_angle, r, f"{r}%", va="center", ha=label_ha,
                    fontsize=10, color="#555555", zorder=30)
        t.set_clip_on(False)

    # Data polygons (fill + line + scatter dots)
    for vals, col_line, col_fill, name in [
        (vals_a_grp, COL_A, FILL_A, a_name),
        (vals_b_grp, COL_B, FILL_B, b_name),
    ]:
        v = vals + vals[:1]
        ax.fill(angles, v, alpha=0.18, color=col_fill, zorder=2)
        ax.plot(angles, v, "-", linewidth=2.0, color=col_line, zorder=4, label=name)
        ax.scatter(angles[:-1], vals, s=50, color=col_line,
                   edgecolors="white", linewidths=1.2, zorder=5)

    # Spoke feature labels with white backing box
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color="#222222")
    ax.tick_params(axis="x", pad=4)
    for txt in ax.get_xticklabels():
        txt.set_zorder(40)
        txt.set_bbox(dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.90))
        txt.set_clip_on(False)
        txt.get_bbox_patch().set_zorder(40)

    ax.set_title(ftype, fontsize=12.5, fontweight="bold", pad=26, color="#1a1a2e", loc="center")

plt.savefig(OUT_DIR / "vox_spider.pdf", bbox_inches="tight", facecolor="white")
print("Saved →", OUT_DIR / "vox_spider.pdf")
