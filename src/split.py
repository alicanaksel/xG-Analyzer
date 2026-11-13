# ==========================================
# Train/test split by match
# ==========================================
#
# Input : data/processed/shots_game_state.csv   (from game_state.py)
# Output: data/processed/train.csv, data/processed/test.csv
#
# Open-play shots only. All shots of one match go to the same side,
# so the model is never tested on a match it has already seen.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(__file__).resolve().parent.parent
IN_PATH = BASE / "data" / "processed" / "shots_game_state.csv"
TRAIN_PATH = BASE / "data" / "processed" / "train.csv"
TEST_PATH = BASE / "data" / "processed" / "test.csv"

TEST_SIZE = 0.2      # share of matches in the test set
RANDOM_STATE = 42    # fixed seed: same split on every run and every computer


def load_open_play(path=IN_PATH):
    """Load the shots table and keep open-play shots only."""
    shots = pd.read_csv(path)
    return shots[shots["shot_type"] == "shot"].reset_index(drop=True)


def split_by_match(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split shots into train and test so that no match is in both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["matchId"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def check_split(train, test):
    """Raise an error if a match is in both sets; return a small summary table."""
    shared = set(train["matchId"]) & set(test["matchId"])
    if shared:
        raise ValueError(f"{len(shared)} matches are in both train and test")

    return pd.DataFrame({
        "matches": [train["matchId"].nunique(), test["matchId"].nunique()],
        "shots": [len(train), len(test)],
        "goals": [train["is_goal"].sum(), test["is_goal"].sum()],
        "goal_rate": [train["is_goal"].mean(), test["is_goal"].mean()],
    }, index=["train", "test"])


if __name__ == "__main__":
    shots = load_open_play()
    train, test = split_by_match(shots)
    summary = check_split(train, test)

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(summary.round(3).to_string())
    print(f"\nSaved to {TRAIN_PATH.name} and {TEST_PATH.name}")
