# Notes

Daily notes and decisions.

## 2026-11-09
- Understat (soccerdata) was down for maintenance, so we switched to the Wyscout event data from Figshare (Pappalardo et al.).
- `events_England.json` was converted to CSV: `data/raw/events_England.csv` (643,150 events, 380 matches).
- Coordinates are on a 0-100 scale (percent of the pitch). They will be rescaled to real pitch size when computing distance/angle.

## 2026-11-11
- `clean.py` keeps only `eventName == "Shot"`. Penalties (80) and direct free-kick shots (350) are stored under `Free Kick`, so they were missing.
- Added `src/extra_shots.py` (reuses the `clean.py` helpers, `clean.py` itself is unchanged). It adds a `shot_type` column: `shot`, `free_kick_shot`, `penalty`. Output: `data/processed/shots_all.csv`.
- Goal rate by type: shot 10.8%, free_kick_shot 5.1%, penalty 70%. Penalties behave very differently, so they should probably be modelled separately.
