import plotly.graph_objects as go
import numpy as np
from IPython.display import HTML, display
from typing import Dict, Any


def visualize_text_interactive(r: Dict[str, Any], show_plot: bool = True):
    """
    Renders an HTML text view (colored by importance).
    Optionally renders a Plotly Bar Chart for token attribution scores.
    """

    def _style(score):
        # Scale intensity (max alpha ~0.1 to keep text readable)
        val = min(abs(score) * 1000, 255)
        color = "255, 0, 0" if score > 0 else "0, 0, 255"
        return f"background-color: rgba({color}, {val/2550:.2f}); padding: 2px 4px; margin: 0 1px; border-radius: 3px;"

    # --- HTML Highlight View ---
    i = 0
    html = f"<div style='border:1px solid #ddd; padding:10px; margin-bottom:10px; border-radius:5px; font-family:monospace;'>"
    html += f"<h4 style='margin:0 0 10px;'>Ex {i+1} | True: {r['true_class']} | Pred: {r.get('pred_class', '?')}</h4>"
    html += " ".join(
        [
            f"<span title='{s:.4f}' style='{_style(s)}'>{t}</span>"
            for t, s in zip(r["tokens"], r["attributions"])
        ]
    )
    html += "</div>"
    display(HTML(html))

    # --- Optional Plotly Bar Chart ---
    if show_plot:
        tokens, attrs = np.array(r["tokens"]), np.array(r["attributions"])
        colors = ["#EF553B" if v > 0 else "#636EFA" for v in attrs]

        fig = go.Figure(go.Bar(x=tokens, y=attrs, marker_color=colors))
        fig.update_layout(
            title=f"Attribution: Example {i+1}",
            margin=dict(l=20, r=20, t=30, b=20),
            height=250,
            template="plotly_white",
            yaxis_title="Score",
        )
        fig.show()
