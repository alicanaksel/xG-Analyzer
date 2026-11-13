# ==========================================
# Game state: score difference at the moment of each shot
# ==========================================
#
# Input : data/raw/events_England.csv      (to find every goal)
#         data/processed/shots_features.csv (from features.py)
# Output: data/processed/shots_game_state.csv
#
# New columns:
#   goals_for, goals_against   score of the shooting team / the opponent BEFORE the shot
#   score_diff                 goals_for - goals_against
#   game_state                 "winning", "drawing" or "losing"

from pathlib import Path

import numpy as np
import pandas as pd

from clean import RAW_PATH, load_events

BASE = Path(__file__).resolve().parent.parent
IN_PATH = BASE / "data" / "processed" / "shots_features.csv"
OUT_PATH = BASE / "data" / "processed" / "shots_game_state.csv"

TAG_GOAL = "101"
TAG_OWN_GOAL = "102"

# Events that can score a goal (Save attempt also has tag 101, but that is the keeper conceding)
SHOT_EVENTS = [("Shot", "Shot"), ("Free Kick", "Penalty"), ("Free Kick", "Free kick shot")]

# One half lasts < 3600 s, so 2H events always get a larger time than 1H events
PERIOD_OFFSET = {"1H": 0, "2H": 10_000}


def match_time(df):
    """One sortable number per event: half offset + seconds in the half."""
    return df["matchPeriod"].map(PERIOD_OFFSET) + df["eventSec"]


def has_tag(tags, tag):
    """True where the pipe separated tag text contains `tag` (exact match)."""
    return tags.fillna("").str.split("|").apply(lambda t: tag in t)


def find_goals(events):
    """Return one row per goal: matchId, scoring teamId, time."""
    is_shot = pd.Series(False, index=events.index)
    for name, sub in SHOT_EVENTS:
        is_shot |= (events["eventName"] == name) & (events["subEventName"] == sub)

    goals = events[is_shot & has_tag(events["tags"], TAG_GOAL)]
    goals = goals[["matchId", "teamId", "matchPeriod", "eventSec"]].copy()

    # Own goals: tagged on the conceding team, so give the goal to the opponent
    own = events[has_tag(events["tags"], TAG_OWN_GOAL)]
    own = own[["matchId", "teamId", "matchPeriod", "eventSec"]].copy()
    own["teamId"] = opponent_of(events, own)

    goals = pd.concat([goals, own], ignore_index=True)
    goals["time"] = match_time(goals)
    return goals[["matchId", "teamId", "time"]]


def opponent_of(events, rows):
    """For each row, the other team in the same match."""
    teams = events.groupby("matchId")["teamId"].unique()  # matchId -> [team_a, team_b]
    return [
        next(t for t in teams[m] if t != team)
        for m, team in zip(rows["matchId"], rows["teamId"])
    ]


def add_game_state(shots, goals):
    """Add goals_for, goals_against, score_diff and game_state to `shots`."""
    shots = shots.copy()
    shots["time"] = match_time(shots)

    # Pair every shot with every goal of the same match, keep goals scored earlier
    pairs = shots[["id", "matchId", "teamId", "time"]].merge(
        goals, on="matchId", suffixes=("", "_goal")
    )
    pairs = pairs[pairs["time_goal"] < pairs["time"]]
    pairs["is_for"] = pairs["teamId_goal"] == pairs["teamId"]

    score = pairs.groupby("id")["is_for"].agg(goals_for="sum", goals_against=lambda s: (~s).sum())

    # Shots with no earlier goal are not in `score`, so they get 0-0
    shots = shots.merge(score, left_on="id", right_index=True, how="left")
    shots[["goals_for", "goals_against"]] = shots[["goals_for", "goals_against"]].fillna(0).astype(int)

    shots["score_diff"] = shots["goals_for"] - shots["goals_against"]
    shots["game_state"] = np.select(
        [shots["score_diff"] > 0, shots["score_diff"] < 0], ["winning", "losing"], default="drawing"
    )
    return shots.drop(columns=["time"])


if __name__ == "__main__":
    events = load_events(RAW_PATH)
    shots = pd.read_csv(IN_PATH)

    goals = find_goals(events)
    shots = add_game_state(shots, goals)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(OUT_PATH, index=False)

    print(f"Goals found: {len(goals)}")
    print(shots["game_state"].value_counts().to_string())
    print(f"\nSaved {len(shots)} shots, {shots.shape[1]} columns to {OUT_PATH}")
