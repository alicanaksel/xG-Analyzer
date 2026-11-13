# xG-Analyzer

An expected goals (xG) model for the 2017/18 Premier League, built on the public Wyscout
event data (Pappalardo et al., Figshare). Work in progress.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The raw data is not in Git (too large). Download `events_England.json` from the Figshare
dataset and put it in `data/raw/`.

## Pipeline

Run the scripts in this order from the project root. Each step reads the output of the previous one.

| Step | Script | Output |
|---|---|---|
| 1. JSON to CSV | `python src/convert_to_csv.py` | `data/raw/events_England.csv` |
| 2. Clean open-play shots | `python src/clean.py` | `data/processed/shots_clean.csv` |
| 3. Add penalties and free-kick shots | `python src/extra_shots.py` | `data/processed/shots_all.csv` |
| 4. Add distance and angle | `python src/features.py` | `data/processed/shots_features.csv` |
| 5. Add game state (score at the moment of the shot) | `python src/game_state.py` | `data/processed/shots_game_state.csv` |
| 6. Train/test split by match (open play, 20% test) | `python src/split.py` | `data/processed/train.csv`, `test.csv` |

Model metrics and calibration helpers are in `src/evaluate.py`.

## Notebooks

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Data checks; why penalties and free kicks were missing and how they were added |
| `02_eda_plots.ipynb` | Shot distribution and goal rate plots |
| `03_features.ipynb` | Distance and angle formulas, tests and analysis |
| `04_game_state.ipynb` | Body part and game state: goal rates and checks |

The raw columns, tag ids and cleaning decisions are documented in `docs/data_dictionary.md`.

Figures are saved to `reports/figures/`. Daily notes and decisions are in `NOTES.md`.

## Columns in `shots_game_state.csv`

| Column | Meaning |
|---|---|
| `playerId`, `teamId`, `matchId`, `id` | Wyscout ids (player, team, match, event) |
| `matchPeriod`, `eventSec` | Half (`1H`/`2H`) and seconds since the start of that half |
| `start_x`, `start_y` | Shot location in percent of the pitch (attacking towards x = 100) |
| `is_goal` | Target: did the shot become a goal? |
| `left_foot`, `right_foot`, `header` | Body part used |
| `counter_attack` | Shot came from a counter attack |
| `shot_type` | `shot` (open play), `free_kick_shot` or `penalty` |
| `x_m`, `y_m` | Shot location in metres (pitch assumed 105 m x 68 m) |
| `distance` | Distance from the shot to the centre of the goal (m) |
| `angle`, `angle_deg` | Angle between the lines to the two goal posts (radians / degrees) |
| `goals_for`, `goals_against` | Goals of the shooting team / the opponent before the shot (own goals included) |
| `score_diff` | `goals_for - goals_against` |
| `game_state` | `winning`, `drawing` or `losing` at the moment of the shot |
