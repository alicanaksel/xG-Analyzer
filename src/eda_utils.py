# ==========================================
# Helper functions for the EDA plots
# ==========================================
#
# Coordinates in the Wyscout data are percentages of the pitch (0-100).
# We attack towards x = 100 (the opponent's goal line); y = 50 is the centre.

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# Real pitch is about 105 m x 68 m. Used only to draw realistic-looking lines.
PITCH_LENGTH_M = 105
PITCH_WIDTH_M = 68


def add_minute_in_half(df):
    """Add `minute` = minutes since the start of the half the shot was taken in.

    `eventSec` restarts at 0 in the second half, so we keep the halves apart
    instead of adding 45 (first-half stoppage time would overlap with the
    start of the second half).
    """
    df = df.copy()
    df["minute"] = df["eventSec"] / 60
    return df


def wilson_interval(goals, n, z=1.96):
    """95% Wilson confidence interval for a goal rate.

    A rate such as "30%" is much less reliable when it comes from 10 shots than
    from 1,000 shots. The interval shows how much the rate could move by chance.
    """
    if n == 0:
        return np.nan, np.nan
    p = goals / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def goal_rate_table(df, group_col):
    """Shots, goals, goal rate and 95% interval for every value of `group_col`."""
    grouped = df.groupby(group_col, observed=True)["is_goal"].agg(shots="size", goals="sum")
    grouped["goal_rate"] = grouped["goals"] / grouped["shots"]
    bounds = [wilson_interval(g, n) for g, n in zip(grouped["goals"], grouped["shots"])]
    grouped["ci_low"] = [b[0] for b in bounds]
    grouped["ci_high"] = [b[1] for b in bounds]
    return grouped


def plot_goal_rate(ax, table, title="", xlabel="", color="#2a6f97"):
    """Bar chart of goal rate with 95% error bars and the shot count above each bar."""
    labels = [str(i) for i in table.index]
    rate = table["goal_rate"].to_numpy()
    err_low = rate - table["ci_low"].to_numpy()
    err_high = table["ci_high"].to_numpy() - rate

    ax.bar(labels, rate * 100, color=color, yerr=[err_low * 100, err_high * 100],
           capsize=4, ecolor="#333333")
    for i, (r, n, hi) in enumerate(zip(rate, table["shots"], table["ci_high"])):
        ax.text(i, hi * 100 + 0.8, f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Goal rate (%)")
    ax.set_ylim(0, max(table["ci_high"].max() * 100 * 1.15, 5))


def draw_half_pitch(ax):
    """Draw the attacking half of the pitch (x from 50 to 100) in 0-100 coordinates."""
    line = {"fill": False, "edgecolor": "#444444", "linewidth": 1.2}

    # Penalty area: 16.5 m deep, 40.32 m wide
    depth = 16.5 / PITCH_LENGTH_M * 100
    half_w = 40.32 / 2 / PITCH_WIDTH_M * 100
    ax.add_patch(Rectangle((100 - depth, 50 - half_w), depth, 2 * half_w, **line))

    # Goal area: 5.5 m deep, 18.32 m wide
    depth_s = 5.5 / PITCH_LENGTH_M * 100
    half_w_s = 18.32 / 2 / PITCH_WIDTH_M * 100
    ax.add_patch(Rectangle((100 - depth_s, 50 - half_w_s), depth_s, 2 * half_w_s, **line))

    # Goal (7.32 m wide), drawn just behind the goal line
    half_goal = 7.32 / 2 / PITCH_WIDTH_M * 100
    ax.add_patch(Rectangle((100, 50 - half_goal), 1.2, 2 * half_goal, **line))

    # Penalty spot (11 m from the goal line)
    ax.plot(100 - 11 / PITCH_LENGTH_M * 100, 50, "o", color="#444444", markersize=3)

    ax.set_xlim(50, 102)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#444444")
    return ax
