# Data Dictionary

Source: Wyscout event data, Pappalardo et al. (2019), Figshare.
File: `data/raw/events_England.json` -> `data/raw/events_England.csv`

Drop reasons used in this file:

- **constant**: same value in every shot row, so the model can learn nothing from it
- **leakage**: only known after the shot, so the model would be cheating
- **replaced**: its useful information was moved into other columns

## 1. Competition choice

The Wyscout dataset covers the 2017/18 season of five leagues (England, France, Germany, Italy, Spain), the 2018 World Cup and Euro 2016.

We use the **Premier League 2017/18**. It is the league we follow most, so we can sanity-check the results (for example xG per player or team) with our own football knowledge.

Size: 380 matches, 8,881 shots (8,451 open play, 350 direct free kicks, 80 penalties), 988 goals.

Wyscout was used instead of Understat because Understat was down for maintenance (see `NOTES.md`, 2026-11-09).

## 2. Raw event columns

| Column | Meaning | Example | Keep? | Reason |
|---|---|---|---|---|
| `id` | Unique id of the event | `177959212` | Yes | ID, used to find a single shot again |
| `eventId` | Event type code (10 = shot) | `10` | Drop | constant |
| `eventName` | Event type name (Pass, Duel, Shot, ...) | `Shot` | Drop | constant (used to filter shots, then dropped) |
| `subEventId` | Sub-type code (100 = shot) | `100` | Drop | constant |
| `subEventName` | Sub-type name | `Shot` | Drop | constant |
| `tags` | Wyscout tag ids, pipe separated (see section 4) | `402\|201\|1215\|1802` | Drop | replaced (useful tags -> bool columns), other tags leak |
| `playerId` | Player who took the shot | `25413` | Yes | ID, needed for xG per player |
| `teamId` | Team of the player | `1609` | Yes | ID, needed for xG per team |
| `matchId` | Match | `2499719` | Yes | ID, needed for xG per match and train/test split by match |
| `matchPeriod` | Half: `1H` or `2H` | `1H` | Yes | Match context |
| `eventSec` | Seconds since the start of the half | `94.6` | Yes | Match context, needed to order events (game state) |
| `start_x`, `start_y` | Shot location (percent of the pitch, see section 3) | `88`, `41` | Yes | Most important input: distance and angle come from here |
| `end_x`, `end_y` | End location of the event | `100`, `100` | Drop | leakage, and for shots it is only 0 or 100 (not a real location) |
| `is_goal` | Added by `convert_to_csv.py`: True if the tags contain `101` | `True` | Yes | Target |

## 3. Coordinates

- `x` goes from 0 (own goal line) to 100 (opponent goal line). The team on the ball always attacks towards x = 100.
- `y` goes from 0 to 100 across the pitch (touchline to touchline). y = 50 is the centre.
- The centre of the opponent goal is at **(100, 50)**.
- Values are percentages, not metres. One unit on x is 1.05 m but one unit on y is only 0.68 m (pitch assumed 105 m x 68 m), so both axes must be converted to metres before computing distance or angle (`src/features.py`).

## 4. Tags used in shots

Counts are from the 8,451 open-play shots.

| Tag id | Count | Meaning | Known at shot time? | Used as |
|---|---|---|---|---|
| `101` | 914 | Goal | No (it is the outcome) | `is_goal` (target) |
| `401` | 2785 | Left foot | Yes | `left_foot` |
| `402` | 4349 | Right foot | Yes | `right_foot` |
| `403` | 1317 | Head or other body part | Yes | `header` |
| `1901` | 425 | Counter attack | Yes | `counter_attack` |
| `201` | 5917 | Opportunity (analyst's judgement, given afterwards) | No | not used |
| `1801` / `1802` | 2986 / 5465 | Accurate / not accurate (on target or not) | No | not used |
| `1201`-`1223` | 6180 | Zone of the goal (or outside it) where the ball went | No | not used |
| `2101` | 2271 | Blocked | No | not used |

Every shot has exactly one of `401`, `402`, `403` (2785 + 4349 + 1317 = 8451).

Watch out: tag `401` is a substring of `1401` (interception). Always split the tags into a list before checking, never search inside the text.

## 5. Missing values

| Column | Missing (all events) | Caused by | Missing in shots |
|---|---|---|---|
| `tags` | 61,481 | Events with no tags (mostly Interruption, Others on the ball, Foul, Free Kick, Offside) | 0 |
| `subEventName`, `subEventId` | 1,558 | Offside events (no sub-type) | 0 |
| `end_x`, `end_y` | 118 | Foul events (no end location) | 0 |

All missing values disappear once we keep only shots. Penalties and free-kick shots (`extra_shots.py`) have no missing values either.

## 6. Useful columns for the model

**Target**

- `is_goal`

**Model inputs**

- `distance`, `angle` (from `start_x`, `start_y`, added by `features.py`)
- `header` (body part; `left_foot` / `right_foot` are kept for analysis)
- `counter_attack`
- `shot_type` (added by `extra_shots.py`; penalties are modelled separately)

**IDs and context (not model inputs)**

- `id`, `playerId`, `teamId`, `matchId`: to group xG by player, team or match
- `matchPeriod`, `eventSec`: to order events in a match
