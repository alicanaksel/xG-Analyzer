# ==========================================
# Convert events_England.json -> CSV
# ==========================================

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EVENTS_PATH = BASE / "data" / "raw" / "events_England.json"
OUT_PATH = BASE / "data" / "raw" / "events_England.csv"

# Wyscout tag id for a goal
TAG_GOAL = 101


def load_events(path=EVENTS_PATH):
    """Load the events JSON into a flat DataFrame."""
    with open(path, encoding="utf-8") as f:
        events = json.load(f)
    df = pd.DataFrame(events)

    # 'positions' is a list of {x, y} dicts: first = start, second = end (if any)
    df["start_x"] = df["positions"].apply(lambda p: p[0]["x"] if len(p) > 0 else None)
    df["start_y"] = df["positions"].apply(lambda p: p[0]["y"] if len(p) > 0 else None)
    df["end_x"] = df["positions"].apply(lambda p: p[1]["x"] if len(p) > 1 else None)
    df["end_y"] = df["positions"].apply(lambda p: p[1]["y"] if len(p) > 1 else None)

    # 'tags' is a list of {id: ...} dicts -> list of ints
    df["tag_ids"] = df["tags"].apply(lambda t: [x["id"] for x in t])
    df["is_goal"] = df["tag_ids"].apply(lambda t: TAG_GOAL in t)
    return df


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading JSON...")
    df = load_events()

    # CSV cannot hold nested lists, so flatten them:
    #  - tags      -> "1801|101"  (pipe separated tag ids)
    #  - positions -> already split into start_x/start_y/end_x/end_y
    df["tags"] = df["tag_ids"].apply(lambda t: "|".join(map(str, t)))
    df = df.drop(columns=["positions", "tag_ids"])

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {OUT_PATH}")
