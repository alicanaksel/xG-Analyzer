# ==========================================
# Extra shots: penalties + direct free-kick shots
# ==========================================
#
# Why this file exists:
#   In the Wyscout data, `clean.py` keeps only events with eventName == "Shot".
#   Penalties and direct free-kick shots are stored under eventName == "Free Kick",
#   so they are missing from shots_clean.csv.
#
# We do NOT touch clean.py. We reuse its helper functions so the output has
# exactly the same columns as shots_clean.csv, plus one new column: `shot_type`.

from pathlib import Path

import pandas as pd

from clean import RAW_PATH, clean_shots, drop_unused_columns, load_events, parse_tags

BASE = Path(__file__).resolve().parent.parent
OUT_PATH = BASE / "data" / "processed" / "shots_all.csv"

# Wyscout subEventName -> our readable shot_type label
EXTRA_SUBEVENTS = {
    "Penalty": "penalty",
    "Free kick shot": "free_kick_shot",
}


def extract_extra_shots(df):
    """Take penalties and direct free-kick shots from the raw events.

    They are cleaned with the same functions clean.py uses, so the columns
    match shots_clean.csv.
    """
    extra = df[
        (df["eventName"] == "Free Kick")
        & (df["subEventName"].isin(EXTRA_SUBEVENTS))
    ].copy()

    # Save the type BEFORE drop_unused_columns removes subEventName
    extra["shot_type"] = extra["subEventName"].map(EXTRA_SUBEVENTS)

    extra = drop_unused_columns(extra)  # same helper as clean.py
    extra = parse_tags(extra)           # same helper as clean.py
    return extra


def build_all_shots(df):
    """Regular shots (clean.py) + penalties + free-kick shots in one table."""
    regular = clean_shots(df)           # clean.py, unchanged
    regular["shot_type"] = "shot"

    extra = extract_extra_shots(df)

    all_shots = pd.concat([regular, extra], ignore_index=True)
    # Keep the original column order, with shot_type last
    return all_shots[list(regular.columns)]


if __name__ == "__main__":
    events = load_events(RAW_PATH)
    shots = build_all_shots(events)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(OUT_PATH, index=False)

    print(shots["shot_type"].value_counts().to_string())
    print(f"\nSaved {len(shots)} shots, {shots.shape[1]} columns to {OUT_PATH}")
