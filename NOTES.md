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

## 2026-11-13 - Days 1-3 follow-up (Hasan)
- Added `docs/data_dictionary.md`: raw columns (keep/drop and why), coordinates, tag ids (known at shot time or not), missing values, useful columns. Competition: Premier League 2017/18, chosen because it is the league we follow most, so results can be checked with football knowledge.
- `clean.py`: added `drop_missing`, `drop_duplicate_events` and `validate_shots` (no missing values, unique ids, coordinates in 0-100, exactly one body part). Real data: nothing removed. Note: `extra_shots.py` calls `drop_unused_columns`/`parse_tags` directly, so these checks do not run on penalties and free kicks yet.
- Added `src/game_state.py`: score before each shot (`goals_for`, `goals_against`, `score_diff`, `game_state`). Goals = Shot / Penalty / Free kick shot with tag 101 (988) + own goals, tag 102, given to the opponent (29) = 1017. Save attempt events also carry tag 101 (keeper conceding) and are not counted.
- Check: Arsenal 4-3 Leicester goal order matches the real match. Against keeper-conceded goals, 758 of 760 match-team rows agree; the 2 others are Wyscout recording errors (a goal with no shot event, a free kick without the goal tag).
- Body part (`16_goal_rate_body_part_by_distance.png`): headers score more overall (13.5% vs 10.3%) only because they are taken closer (median 10.3 m vs 18.9 m). At the same distance headers score less (0-8 m: 27% vs 46%). Distance is a confounder, so `header` must be used together with `distance`.
- Game state (`17_goal_rate_game_state.png`, open play): losing 9.3%, drawing 10.0%, winning 13.8%. The gap stays at the same distance (0-8 m: 32.8% / 33.8% / 46.7%). Winning teams shoot more from counter attacks (7.5% vs 3.3%) and head less (10.8% vs 19.7%).
- Interpretation: the losing team pushes forward and leaves space behind, so the leading team gets cleaner chances (fits the higher counter-attack share). Psychology also plays a role: the leading team finishes more calmly and confidently (cannot be measured with this data).
- Decision for Day 4: build the baseline without game state first, then add it and check whether the model improves.
