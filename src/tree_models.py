# ==========================================
# Tree models: Random Forest and XGBoost (Day 5)
# ==========================================
#
# Input : data/processed/train.csv, data/processed/test.csv   (from split.py)
# Output: data/processed/test_predictions.csv                  (xG of every model per test shot)
#
# Same rules as the logistic regression (model.py):
#   - same features (model A), so the only difference is the type of model
#   - hyperparameters chosen with 5-fold GroupKFold by match on the train set,
#     scored with log loss (xG is a probability, so accuracy is the wrong target)
#   - the test set is used once, at the end
#
# Run from the project root:
#   python src/tree_models.py          # uses the tuned settings below (fast)
#   python src/tree_models.py --tune   # runs the grid searches again (a few minutes)
#
# macOS: XGBoost needs the OpenMP runtime -> brew install libomp

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from xgboost import XGBClassifier

from evaluate import baseline_prob, compare_models, evaluate
from model import FEATURES_A, FEATURES_B, TARGET, fit_logit, load_train_test, predict_xg

BASE = Path(__file__).resolve().parent.parent
PRED_PATH = BASE / "data" / "processed" / "test_predictions.csv"

RANDOM_STATE = 42
N_SPLITS = 5

# ---- Search grids ----------------------------------------------------------
# Random Forest: the main danger is overfitting (leaves with very few shots
# give probabilities of 0 or 1, which log loss punishes hard), so we limit
# the depth and force a minimum number of shots per leaf.
RF_GRID = {
    "max_depth": [4, 5, 6, 8],
    "min_samples_leaf": [20, 35, 50, 100],
    "max_features": [1, 2],
}

# XGBoost: many small trees, each correcting the previous ones.
# max_depth = 1 means no interactions between features (each tree uses one feature).
XGB_GRID = {
    "max_depth": [1, 2, 3],
    "learning_rate": [0.01, 0.02, 0.05],
    "n_estimators": [150, 300, 600],
    "min_child_weight": [1, 10],
}

# ---- Best settings found with the grids above (5-fold GroupKFold, log loss) ----
RF_BEST = {"n_estimators": 500, "max_depth": 5, "min_samples_leaf": 20, "max_features": 2}
XGB_BEST = {"n_estimators": 600, "learning_rate": 0.02, "max_depth": 1, "min_child_weight": 10}
XGB_MONO_BEST = {"n_estimators": 600, "learning_rate": 0.01, "max_depth": 2, "min_child_weight": 10}

# Football logic the model must follow: further away never raises xG,
# a wider angle never lowers it. 0 = no constraint.
MONOTONE_DIRECTION = {"distance": -1, "angle": 1}


def monotone_constraints(features):
    """Constraint tuple for XGBoost, in the same order as `features`."""
    return tuple(MONOTONE_DIRECTION.get(f, 0) for f in features)


def make_random_forest(**params):
    """Random Forest with fixed seed; `params` overrides the defaults."""
    settings = {"n_estimators": 500, "random_state": RANDOM_STATE, "n_jobs": -1}
    settings.update(params)
    return RandomForestClassifier(**settings)


def make_xgboost(features, monotone=False, **params):
    """XGBoost classifier. With monotone=True, distance/angle follow MONOTONE_DIRECTION."""
    settings = {"subsample": 0.8, "random_state": RANDOM_STATE, "n_jobs": 4, "eval_metric": "logloss"}
    if monotone:
        settings["monotone_constraints"] = monotone_constraints(features)
    settings.update(params)
    return XGBClassifier(**settings)


def feature_matrix(df, features):
    """Features as floats (True/False -> 1/0)."""
    return df[features].astype(float)


def tune(estimator, grid, train, features, n_splits=N_SPLITS):
    """Grid search with GroupKFold by match, scored by log loss.

    Returns the fitted search and a results table sorted from best to worst, with the
    train log loss next to the validation log loss (a big gap = overfitting).
    """
    search = GridSearchCV(estimator, grid, scoring="neg_log_loss", cv=GroupKFold(n_splits=n_splits),
                          return_train_score=True)
    search.fit(feature_matrix(train, features), train[TARGET].astype(int), groups=train["matchId"])

    results = pd.DataFrame(search.cv_results_)
    table = results[[f"param_{p}" for p in grid]].copy()
    table.columns = list(grid)
    table["cv_log_loss"] = -results["mean_test_score"]
    table["cv_sd"] = results["std_test_score"]
    table["train_log_loss"] = -results["mean_train_score"]
    table["gap"] = table["cv_log_loss"] - table["train_log_loss"]
    return search, table.sort_values("cv_log_loss").reset_index(drop=True)


def cross_validate_estimator(estimator, train, features, n_splits=N_SPLITS):
    """Per-fold metrics with the same GroupKFold folds as model.cross_validate."""
    rows = []
    folds = GroupKFold(n_splits=n_splits).split(train, groups=train["matchId"])
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        fit_part, val_part = train.iloc[fit_idx], train.iloc[val_idx]
        estimator.fit(feature_matrix(fit_part, features), fit_part[TARGET].astype(int))
        xg = estimator.predict_proba(feature_matrix(val_part, features))[:, 1]
        rows.append({"fold": fold, **evaluate(val_part[TARGET], xg)})
    return pd.DataFrame(rows).set_index("fold")


def fit_predict(estimator, train, test, features):
    """Fit on the full train set and return xG for every test shot."""
    estimator.fit(feature_matrix(train, features), train[TARGET].astype(int))
    return estimator.predict_proba(feature_matrix(test, features))[:, 1]


def bootstrap_difference(y_true, xg_ref, xg_new, groups, metric="log_loss", n_boot=2000, seed=RANDOM_STATE):
    """Bootstrap the difference metric(new) - metric(ref) by resampling whole matches.

    Matches (not shots) are resampled because shots of one match are not independent.
    Returns the observed difference and its 95% interval. If the interval contains 0,
    the two models are not clearly different on this test set.
    """
    y_true = np.asarray(y_true, dtype=int)
    xg_ref, xg_new = np.asarray(xg_ref, dtype=float), np.asarray(xg_new, dtype=float)
    groups = np.asarray(groups)
    match_ids = np.unique(groups)
    rows_of = {m: np.flatnonzero(groups == m) for m in match_ids}

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sample = rng.choice(match_ids, size=len(match_ids), replace=True)
        idx = np.concatenate([rows_of[m] for m in sample])
        if y_true[idx].min() == y_true[idx].max():
            continue  # a sample without goals (or without misses) cannot be scored
        diffs.append(evaluate(y_true[idx], xg_new[idx])[metric] - evaluate(y_true[idx], xg_ref[idx])[metric])

    observed = evaluate(y_true, xg_new)[metric] - evaluate(y_true, xg_ref)[metric]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"difference": observed, "ci_low": low, "ci_high": high}


def all_test_predictions(train, test):
    """xG of every model (baseline, logistic A/B, RF, XGBoost, XGBoost monotone) on the test set."""
    preds = test[["id", "matchId", "teamId", "playerId", TARGET]].copy()
    preds["xg_baseline"] = baseline_prob(train[TARGET], len(test))
    preds["xg_logit_A"] = predict_xg(fit_logit(train, FEATURES_A), test, FEATURES_A)
    preds["xg_logit_B"] = predict_xg(fit_logit(train, FEATURES_B), test, FEATURES_B)
    preds["xg_rf"] = fit_predict(make_random_forest(**RF_BEST), train, test, FEATURES_A)
    preds["xg_xgb"] = fit_predict(make_xgboost(FEATURES_A, **XGB_BEST), train, test, FEATURES_A)
    preds["xg_xgb_mono"] = fit_predict(make_xgboost(FEATURES_A, monotone=True, **XGB_MONO_BEST),
                                       train, test, FEATURES_A)
    return preds


if __name__ == "__main__":
    train, test = load_train_test()

    if "--tune" in sys.argv:
        for name, est, grid in [
            ("Random Forest", make_random_forest(), RF_GRID),
            ("XGBoost", make_xgboost(FEATURES_A), XGB_GRID),
            ("XGBoost monotone", make_xgboost(FEATURES_A, monotone=True), XGB_GRID),
        ]:
            search, table = tune(est, grid, train, FEATURES_A)
            print(f"\n{name} - best: {search.best_params_}")
            print(table.head(5).round(4).to_string())

    preds = all_test_predictions(train, test)
    xg_columns = [c for c in preds.columns if c.startswith("xg_")]
    print("\nTest set:")
    print(compare_models({c[3:]: evaluate(preds[TARGET], preds[c]) for c in xg_columns}).round(4).to_string())

    preds.to_csv(PRED_PATH, index=False)
    print(f"\nSaved {len(xg_columns)} models' predictions to {PRED_PATH}")
