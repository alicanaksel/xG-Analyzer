# ==========================================
# Geometry features: distance and angle to goal
# ==========================================
#
# Input : data/processed/shots_all.csv   (from extra_shots.py)
# Output: data/processed/shots_features.csv
#
# Wyscout coordinates are percentages of the pitch (0-100):
#   x = 0 own goal line, x = 100 opponent goal line (we always attack to the right)
#   y = 0 and y = 100 are the touchlines, y = 50 is the centre
#
# New columns:
#   x_m, y_m        shot location in metres (pitch assumed 105 m x 68 m)
#   distance        straight-line distance from the shot to the centre of the goal (m)
#   angle           angle between the lines to the two goal posts (radians)
#   angle_deg       same angle in degrees (easier to read)
#
# Run from the project root:  python src/features.py

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
IN_PATH = BASE / "data" / "processed" / "shots_all.csv"
OUT_PATH = BASE / "data" / "processed" / "shots_features.csv"

# Standard pitch and goal size in metres. Real pitches vary a little per
# stadium; the data does not tell us the real size, so we use the standard.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32

# Goal position in metres
GOAL_X = PITCH_LENGTH
GOAL_Y = PITCH_WIDTH / 2
POST_LEFT_Y = GOAL_Y - GOAL_WIDTH / 2
POST_RIGHT_Y = GOAL_Y + GOAL_WIDTH / 2


def to_meters(x_pct, y_pct):
    """Convert Wyscout percentage coordinates to metres.

    One percent on the x axis is 1.05 m, but on the y axis it is only 0.68 m,
    so the two axes must be scaled separately before any distance is computed.
    """
    x_m = np.asarray(x_pct, dtype=float) / 100 * PITCH_LENGTH
    y_m = np.asarray(y_pct, dtype=float) / 100 * PITCH_WIDTH
    return x_m, y_m


def distance_to_goal(x_m, y_m):
    """Straight-line distance (m) from the shot to the centre of the goal line.

    Pythagoras: sqrt(dx^2 + dy^2), where dx/dy are the gaps to the goal centre.
    """
    dx = GOAL_X - np.asarray(x_m, dtype=float)
    dy = GOAL_Y - np.asarray(y_m, dtype=float)
    return np.hypot(dx, dy)


def shot_angle(x_m, y_m):
    """Angle (radians) the goal mouth covers when seen from the shot location.

    We draw one vector from the shot to each post and measure the angle between
    them. This gives the same result as the law of cosines,
        cos(angle) = (d1^2 + d2^2 - goal_width^2) / (2 * d1 * d2),
    but uses atan2(|cross|, dot), which never divides by zero and stays
    accurate for shots very close to the goal line.

    Values: 0 = the goal is invisible (on the goal line outside the posts),
            pi (180 deg) = standing on the goal line between the posts.
    """
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)

    # Vectors from the shot to the left and right post
    ax, ay = GOAL_X - x_m, POST_LEFT_Y - y_m
    bx, by = GOAL_X - x_m, POST_RIGHT_Y - y_m

    cross = ax * by - ay * bx
    dot = ax * bx + ay * by
    return np.arctan2(np.abs(cross), dot)


def shot_angle_cosine_rule(x_m, y_m):
    """Same angle with the law of cosines. Only used to double-check shot_angle."""
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    d1 = np.hypot(GOAL_X - x_m, POST_LEFT_Y - y_m)
    d2 = np.hypot(GOAL_X - x_m, POST_RIGHT_Y - y_m)
    cos_angle = (d1**2 + d2**2 - GOAL_WIDTH**2) / (2 * d1 * d2)
    return np.arccos(np.clip(cos_angle, -1, 1))


def add_geometry_features(df):
    """Return a copy of `df` with x_m, y_m, distance, angle and angle_deg added."""
    df = df.copy()
    df["x_m"], df["y_m"] = to_meters(df["start_x"], df["start_y"])
    df["distance"] = distance_to_goal(df["x_m"], df["y_m"])
    df["angle"] = shot_angle(df["x_m"], df["y_m"])
    df["angle_deg"] = np.degrees(df["angle"])
    return df


if __name__ == "__main__":
    shots = pd.read_csv(IN_PATH)
    shots = add_geometry_features(shots)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(OUT_PATH, index=False)

    print(shots[["distance", "angle_deg"]].describe().round(2).to_string())
    print(f"\nSaved {len(shots)} shots, {shots.shape[1]} columns to {OUT_PATH}")
