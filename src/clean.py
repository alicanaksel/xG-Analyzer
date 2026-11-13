# ==========================================
# Clean events_England.csv -> shots_clean.csv
# ==========================================

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW_PATH = BASE / "data" / "raw" / "events_England.csv"
OUT_PATH = BASE / "data" / "processed" / "shots_clean.csv"

# Constant columns + end coords (only 0/100 for shots, and known after the shot)
UNUSED_COLUMNS = ["eventName", "eventId", "subEventName", "subEventId", "end_x", "end_y"]

# A shot is unusable without these
REQUIRED_COLUMNS = ["id", "matchId", "teamId", "playerId", "matchPeriod", "eventSec",
                    "start_x", "start_y", "tags", "is_goal"]

# Wyscout tag ids known at the moment of the shot
TAG_LEFT_FOOT = "401"
TAG_RIGHT_FOOT = "402"
TAG_HEADER = "403"
TAG_COUNTER = "1901"


def load_events(path=RAW_PATH):
    """Load the raw events CSV."""
    return pd.read_csv(path)


def filter_shots(df):
    """Keep only shot events."""
    return df[df["eventName"] == "Shot"].copy()


def drop_unused_columns(df):
    """Drop constant and leaking columns, reset the index."""
    return df.drop(columns=UNUSED_COLUMNS).reset_index(drop=True)


def drop_missing(df):
    """Drop rows with a missing value in any required column."""
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if len(df) < before:
        print(f"drop_missing: removed {before - len(df)} rows")
    return df.reset_index(drop=True)


def drop_duplicate_events(df):
    """Keep one row per event id."""
    before = len(df)
    df = df.drop_duplicates(subset="id")
    if len(df) < before:
        print(f"drop_duplicate_events: removed {before - len(df)} rows")
    return df.reset_index(drop=True)


def parse_tags(df):
    """Turn useful tags into bool columns and drop the raw tags column.

    Other tags (goal, accuracy, goal zone, blocked) leak the outcome.
    """
    df = df.copy()
    # Split into a list so "401" does not match inside "1401"
    tag_list = df["tags"].str.split("|")

    df["left_foot"] = tag_list.apply(lambda t: TAG_LEFT_FOOT in t)
    df["right_foot"] = tag_list.apply(lambda t: TAG_RIGHT_FOOT in t)
    df["header"] = tag_list.apply(lambda t: TAG_HEADER in t)
    df["counter_attack"] = tag_list.apply(lambda t: TAG_COUNTER in t)

    return df.drop(columns=["tags"])


def validate_shots(df):
    """Raise an error if the clean table breaks a basic rule."""
    problems = []

    if df.isna().any().any():
        problems.append("missing values left")
    if df["id"].duplicated().any():
        problems.append("duplicate event ids")
    if not (df["start_x"].between(0, 100).all() and df["start_y"].between(0, 100).all()):
        problems.append("coordinates outside 0-100")
    # Every shot needs exactly one body part
    if not (df[["left_foot", "right_foot", "header"]].sum(axis=1) == 1).all():
        problems.append("shots without exactly one body part")

    if problems:
        raise ValueError("validate_shots failed: " + ", ".join(problems))
    return df


def clean_shots(df):
    """Full cleaning pipeline: raw events -> clean shots table."""
    shots = filter_shots(df)
    shots = drop_unused_columns(shots)
    shots = drop_missing(shots)  # before parse_tags: a missing tag would crash it
    shots = drop_duplicate_events(shots)
    shots = parse_tags(shots)
    return validate_shots(shots)


if __name__ == "__main__":
    events = load_events()
    shots = clean_shots(events)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(OUT_PATH, index=False)

    print(f"{len(events)} events -> {len(shots)} shots, {shots.shape[1]} columns")
    print(f"Saved to {OUT_PATH}")
