# ==========================================
# Evaluation helpers for xG models
# ==========================================
#
# Every function takes the true outcome (y_true: 1 = goal, 0 = no goal)
# and the model's goal probability for each shot (y_prob, between 0 and 1).
#
# Metrics:
#   log_loss     lower is better; punishes confident wrong predictions hard
#   brier        lower is better; mean squared error of the probabilities
#   roc_auc      higher is better (0.5 = coin flip, 1 = perfect ranking)
#   xg / goals   sum of probabilities vs real goals; close to 1 = well calibrated in total

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def evaluate(y_true, y_prob):
    """Return the main metrics for one model as a dict."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "shots": len(y_true),
        "goals": int(y_true.sum()),
        "xg": y_prob.sum(),
        "xg_per_goal": y_prob.sum() / y_true.sum(),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def baseline_prob(y_train, n):
    """Naive model: every shot gets the train goal rate. A real model must beat this."""
    return np.full(n, np.mean(y_train))


def compare_models(results):
    """Turn {model name: evaluate(...) dict} into one table, one row per model."""
    return pd.DataFrame(results).T


def calibration_table(y_true, y_prob, n_bins=10):
    """Group shots by predicted xG and compare the mean prediction with the real goal rate.

    Bins have equal numbers of shots (quantiles), so every bin has enough shots.
    """
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "p": np.asarray(y_prob, dtype=float)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(
        shots=("y", "size"), mean_xg=("p", "mean"), goal_rate=("y", "mean")
    ).reset_index(drop=True)


def plot_calibration(ax, table, label="model", color="#2a78d6"):
    """Calibration plot: predicted xG (x) vs real goal rate (y). Perfect = on the diagonal."""
    top = max(table["mean_xg"].max(), table["goal_rate"].max()) * 1.05
    ax.plot([0, top], [0, top], linestyle="--", color="#999999", linewidth=1, label="perfect")
    ax.plot(table["mean_xg"], table["goal_rate"], marker="o", color=color, linewidth=2, label=label)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Mean predicted xG")
    ax.set_ylabel("Real goal rate")
    ax.legend(frameon=False)
