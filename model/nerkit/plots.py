"""Small, consistent charts for the notebooks."""

import matplotlib.pyplot as plt

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
TEXT = "#52514e"
GRID = "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor": "#0b0b0b", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": TEXT, "ytick.color": TEXT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def scores_chart(summary, title, column="mix_f1", label="F1, competition mix"):
    """Horizontal bars, best first, with the spread across seeds when there is one."""
    ordered = summary.sort_values(column)
    figure, axis = plt.subplots(figsize=(8, max(2.2, 0.36 * len(ordered) + 1)))
    errors = None
    if "spread" in ordered and column == "mix_f1" and ordered["spread"].max() > 0:
        errors = ordered["spread"] / 2
    axis.barh(ordered.index, ordered[column], height=0.55, color=SERIES[0], xerr=errors,
              error_kw=dict(ecolor=TEXT, elinewidth=1, capsize=3))
    for position, value in enumerate(ordered[column]):
        axis.annotate(f"{value:.3f}", (value, position), xytext=(6, 0), textcoords="offset points",
                      va="center", color=TEXT, fontsize=9)
    axis.grid(axis="y", visible=False)
    axis.set(xlabel=label, title=title)
    axis.margins(x=0.12)
    figure.tight_layout()
    return figure


def learning_curves(records, metric="val_f1", title=None, ylabel="Inner validation F1"):
    """One line per run, over epochs. At most eight runs, each with a fixed color."""
    figure, axis = plt.subplots(figsize=(8, 3.6))
    for color, record in zip(SERIES, records[:len(SERIES)]):
        history = record["history"]
        axis.plot([h["epoch"] for h in history], [h[metric] for h in history], color=color,
                  linewidth=2, marker="o", markersize=5, label=record["name"])
    axis.set(xlabel="Epoch", ylabel=ylabel, title=title or metric)
    axis.xaxis.get_major_locator().set_params(integer=True)
    axis.grid(axis="x", visible=False)
    axis.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    figure.tight_layout()
    return figure


def train_vs_validation(record):
    """Overfitting check: training and validation loss on one axis."""
    figure, axis = plt.subplots(figsize=(6.5, 3.2))
    history = record["history"]
    epochs = [h["epoch"] for h in history]
    axis.plot(epochs, [h["train_loss"] for h in history], color=SERIES[0], linewidth=2,
              marker="o", markersize=5, label="Training")
    axis.plot(epochs, [h["val_loss"] for h in history], color=SERIES[1], linewidth=2,
              marker="o", markersize=5, label="Inner validation")
    axis.set(xlabel="Epoch", ylabel="Loss", title=f"Loss: {record['name']}")
    axis.xaxis.get_major_locator().set_params(integer=True)
    axis.grid(axis="x", visible=False)
    axis.legend()
    figure.tight_layout()
    return figure


def gradient_flow(record, layers=8):
    """Mean absolute gradient per layer. Flat lines near zero mean vanishing gradients."""
    records = record["gradients"]
    figure, axis = plt.subplots(figsize=(8, 3.6))
    names = list(records[0]["gradients"]) if records else []
    for color, name in zip(SERIES, names[:layers]):
        steps = [r["step"] for r in records if name in r["gradients"]]
        values = [max(r["gradients"][name], 1e-12) for r in records if name in r["gradients"]]
        axis.plot(steps, values, color=color, linewidth=2, label=name)
    axis.set(yscale="log", xlabel="Training step", ylabel="Mean |gradient|",
             title=f"Gradient flow: {record['name']}")
    axis.grid(axis="x", visible=False)
    axis.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    figure.tight_layout()
    return figure


def threshold_curve(table, column="threshold", label="Threshold on 1 - P(O)"):
    """Precision, recall and F1 across a decoding knob (a threshold, or a bias for a CRF)."""
    figure, axis = plt.subplots(figsize=(6.5, 3.4))
    for color, name in zip(SERIES, ("f1", "precision", "recall")):
        axis.plot(table[column], table[name], color=color, linewidth=2, label=name)
    axis.set(xlabel=label, ylabel="Score", title="Decoding knob, inner validation")
    axis.grid(axis="x", visible=False)
    axis.legend()
    figure.tight_layout()
    return figure
