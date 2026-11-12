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
- EDA plots (`notebooks/02_eda_plots.ipynb`) use open-play shots only; penalties and free kicks are shown in their own plots.

## 2026-11-12 - Day 3: distance and angle (Ali)
- Added `src/features.py` and `notebooks/03_features.ipynb`. Output: `data/processed/shots_features.csv` (8,881 shots, all shot types).
- New columns: `x_m`, `y_m` (metres, pitch assumed 105 x 68 m), `distance` (m to the goal centre), `angle` (radians between the two posts), `angle_deg`.
- Angle is computed with vectors (`atan2(|cross|, dot)`); it matches the law of cosines on every shot (max difference ~1e-13 deg) and never divides by zero.
- Tests passed: penalty spot 11 m / 36.8 deg, goal line centre 0 m / 180 deg, outside the posts on the goal line 0 deg.
- Penalties in the data: median 10.6 m, 38.1 deg (recorded at x ~ 90, slightly closer than the real 11 m spot).
- Open play: median shot 17.0 m and 19.3 deg. Goal rate 50.3% under 6 m, ~2-3% beyond 20 m; 36.5% for angles above 45 deg.
- Distance and angle are strongly correlated (Pearson -0.75, Spearman -0.89), but at the same distance a wider angle still scores more, so both should go into the model.
- One shot is at x = 100 between the posts (angle 180 deg, a goal). Kept, it is physically possible.
- Decision for Day 4: train the model on open-play shots with `distance`, `angle`, `header`, `counter_attack`; read the distance/angle coefficients together because of the correlation.
