# ==========================================
# Baseline xG model: logistic regression (Day 4)
# ==========================================
#
# Input : data/processed/train.csv, data/processed/test.csv   (from split.py)
# Output: data/processed/test_predictions.csv                  (xG per test shot)
#
# Logistic regression turns a linear score into a probability:
#     z  = b0 + b1 * distance + b2 * angle + ...
#     xG = 1 / (1 + exp(-z))
# exp(coefficient) is the odds ratio: how the odds of a goal change when the
# feature goes up by one unit (1 m, 1 radian, or from 0 to 1 for yes/no features).
#
# We fit with statsmodels because it also gives standard errors, p-values and
# confidence intervals, which we need to interpret the coefficients. The fit is
# plain maximum likelihood (no regularisation), the same as scikit-learn's
# LogisticRegression without a penalty.
#
# Run from the project root:  python src/model.py

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

from evaluate import baseline_prob, compare_models, evaluate

BASE = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE / "data" / "processed" / "train.csv"
TEST_PATH = BASE / "data" / "processed" / "test.csv"
PRED_PATH = BASE / "data" / "processed" / "test_predictions.csv"

TARGET = "is_goal"

# Model A: shot quality only (location, body part, type of attack)
FEATURES_A = ["distance", "angle", "header", "counter_attack"]
# Model B: model A + the score at the moment of the shot
FEATURES_B = FEATURES_A + ["score_diff"]

MODELS = {"A": FEATURES_A, "B": FEATURES_B}


def load_train_test(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """Load the train and test sets made by split.py (open-play shots, split by match)."""
    return pd.read_csv(train_path), pd.read_csv(test_path)


def design_matrix(df, features):
    """Feature matrix as floats (True/False -> 1/0) with an intercept column `const`."""
    return sm.add_constant(df[features].astype(float), has_constant="add")


def fit_logit(df, features):
    """Fit a logistic regression of is_goal on `features`. Returns a statsmodels result."""
    y = df[TARGET].astype(int)
    return sm.Logit(y, design_matrix(df, features)).fit(disp=0)


def predict_xg(result, df, features):
    """xG (goal probability) for every shot in `df`."""
    return np.asarray(result.predict(design_matrix(df, features)))


def coefficient_table(result, alpha=0.05):
    """One row per coefficient: estimate, 95% interval, p-value and odds ratio.

    odds_ratio > 1: the feature raises the chance of a goal
    odds_ratio < 1: the feature lowers it
    If the interval of the odds ratio contains 1, the effect is not clearly different from zero.
    """
    ci = result.conf_int(alpha=alpha)
    table = pd.DataFrame({
        "coef": result.params,
        "std_err": result.bse,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": result.pvalues,
    })
    table["odds_ratio"] = np.exp(table["coef"])
    table["or_low"] = np.exp(table["ci_low"])
    table["or_high"] = np.exp(table["ci_high"])
    return table


def standardize(train, other, columns):
    """Scale `columns` to mean 0 and standard deviation 1, using the train values only.

    Returns scaled copies of both tables and the train standard deviations.
    Yes/no features are left as 0/1 so their coefficients keep the same meaning.
    """
    mean = train[columns].mean()
    std = train[columns].std()
    train_s, other_s = train.copy(), other.copy()
    train_s[columns] = (train[columns] - mean) / std
    other_s[columns] = (other[columns] - mean) / std
    return train_s, other_s, std


def cross_validate(df, features, n_splits=5):
    """Cross-validation on the train set, split by match (GroupKFold).

    Every fold leaves out about 1/n_splits of the matches, fits on the rest and
    scores the left-out matches. Returns one row of metrics per fold.
    Use this to compare models; keep the test set for the final check only.
    """
    rows = []
    folds = GroupKFold(n_splits=n_splits).split(df, groups=df["matchId"])
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        fit_part, val_part = df.iloc[fit_idx], df.iloc[val_idx]
        result = fit_logit(fit_part, features)
        metrics = evaluate(val_part[TARGET], predict_xg(result, val_part, features))
        rows.append({"fold": fold, **metrics})
    return pd.DataFrame(rows).set_index("fold")


def cross_validate_baseline(df, n_splits=5):
    """Same folds as cross_validate, for the 'every shot gets the average' baseline."""
    rows = []
    folds = GroupKFold(n_splits=n_splits).split(df, groups=df["matchId"])
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        y_fit, y_val = df.iloc[fit_idx][TARGET], df.iloc[val_idx][TARGET]
        rows.append({"fold": fold, **evaluate(y_val, baseline_prob(y_fit, len(y_val)))})
    return pd.DataFrame(rows).set_index("fold")


if __name__ == "__main__":
    train, test = load_train_test()

    results = {"baseline": evaluate(test[TARGET], baseline_prob(train[TARGET], len(test)))}
    predictions = test[["id", "matchId", "teamId", "playerId", TARGET]].copy()
    predictions["xg_baseline"] = baseline_prob(train[TARGET], len(test))

    for name, features in MODELS.items():
        fitted = fit_logit(train, features)
        xg = predict_xg(fitted, test, features)
        results[f"logit_{name}"] = evaluate(test[TARGET], xg)
        predictions[f"xg_logit_{name}"] = xg
        print(f"\nModel {name}: {features}")
        print(coefficient_table(fitted).round(3).to_string())

    print("\nTest set:")
    print(compare_models(results).round(4).to_string())

    predictions.to_csv(PRED_PATH, index=False)
    print(f"\nSaved test predictions to {PRED_PATH}")
