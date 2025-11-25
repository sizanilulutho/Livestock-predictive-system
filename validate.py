# src/validate.py
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
FIG_DIR = PROCESSED / "figures"
PROCESSED.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    # Manual sqrt to stay compatible across sklearn versions
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def time_folds_from_years(years: np.ndarray, train_end_starts: list[int], test_len: int = 2):
    """
    Build non-overlapping, forward-time folds:
      - Train: years <= T
      - Test : years in (T, T+test_len]
    `train_end_starts` is a list of T values (end of train window).
    """
    folds = []
    for t_end in train_end_starts:
        test_start = t_end + 1
        test_end = t_end + test_len
        folds.append((t_end, (test_start, test_end)))
    return folds


def pick_feature_columns(df: pd.DataFrame):
    """
    Numeric climate features (annual/sum) + categoricals.
    """
    num_feats = [c for c in df.columns if c.endswith("_ann") or c.endswith("_sum")]
    cat_feats = ["item", "district"]
    return num_feats, cat_feats


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df = pd.read_csv(PROCESSED / "merged.csv")
if "year" not in df.columns or "value" not in df.columns:
    raise ValueError("merged.csv must contain columns: 'year' and 'value'.")

# Ensure types
df["year"] = df["year"].astype(int)

# Feature columns
num_feats, cat_feats = pick_feature_columns(df)
X = df[num_feats + cat_feats].copy()
y = df["value"].copy()

# Preprocessor and model
pre = ColumnTransformer(
    [
        ("num", "passthrough", num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ]
)
rf = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
)
pipe = Pipeline([("pre", pre), ("rf", rf)])


# ---------------------------------------------------------------------
# Build folds across time
# ---------------------------------------------------------------------
min_year, max_year = df["year"].min(), df["year"].max()
# Example: train up to 2012, 2013, 2014, 2015, 2016, 2017; test = next 2 years each time.
train_end_years = [y for y in range(min_year + 12, 2017 + 1)]
folds = time_folds_from_years(df["year"].values, train_end_years, test_len=2)
if not folds:
    raise ValueError("No time folds constructed. Check your year span in merged.csv.")

# ---------------------------------------------------------------------
# Cross-validated evaluation across folds
# ---------------------------------------------------------------------
rows = []
pred_rows = []

for train_end, (t_start, t_end) in folds:
    train_mask = df["year"] <= train_end
    test_mask = (df["year"] >= t_start) & (df["year"] <= t_end)

    # Skip if no test rows
    if test_mask.sum() == 0:
        continue

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    fold_mae = mean_absolute_error(y_test, y_pred)
    fold_rmse = rmse(y_test, y_pred)
    fold_r2 = r2_score(y_test, y_pred)

    rows.append(
        {
            "train_end": train_end,
            "test_start": t_start,
            "test_end": t_end,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "MAE": fold_mae,
            "RMSE": fold_rmse,
            "R2": fold_r2,
        }
    )

    # Store predictions for later inspection
    tmp = df.loc[test_mask, ["year", "item", "district", "value"]].copy()
    tmp["pred"] = y_pred
    tmp["train_end"] = train_end
    tmp["test_window"] = f"{t_start}-{t_end}"
    pred_rows.append(tmp)

cv_df = pd.DataFrame(rows).sort_values(["train_end"]).reset_index(drop=True)
pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

# Save metrics
cv_df.to_csv(PROCESSED / "cv_metrics.csv", index=False)
pred_df.to_csv(PROCESSED / "fold_predictions.csv", index=False)

# Print summary
print("Time-based CV across folds")
print(cv_df)
if not cv_df.empty:
    print("\nAverage over folds:")
    print(
        cv_df[["MAE", "RMSE", "R2"]].mean().rename("mean").to_frame().T.round(3)
    )

# Plot metrics per fold
if not cv_df.empty:
    fig, ax = plt.subplots()
    ax.plot(cv_df["train_end"], cv_df["MAE"], marker="o", label="MAE")
    ax.plot(cv_df["train_end"], cv_df["RMSE"], marker="o", label="RMSE")
    ax2 = ax.twinx()
    ax2.plot(cv_df["train_end"], cv_df["R2"], marker="o", linestyle="--", label="R2")
    ax.set_xlabel("Train end year")
    ax.set_title("Validation metrics by fold (time-based)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "validation_metrics_by_fold.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# Permutation importance on the last fold (fixed)
#   Fit on last training window, transform X_test, run PI on final estimator.
# ---------------------------------------------------------------------
last_train_end, (ts, te) = folds[-1]
mask_test = (df["year"] >= ts) & (df["year"] <= te)
mask_train_last = df["year"] <= last_train_end

if mask_test.sum() > 0:
    pipe.fit(X[mask_train_last], y[mask_train_last])

    preproc = pipe.named_steps["pre"]
    est = pipe.named_steps["rf"]

    Xt_test = preproc.transform(X[mask_test])
    feat_names = list(preproc.get_feature_names_out())

    result = permutation_importance(
        est,
        Xt_test,
        y[mask_test].to_numpy(),
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame(
        {"feature": feat_names, "importance": result.importances_mean}
    ).sort_values("importance", ascending=False)

    imp_df.to_csv(PROCESSED / "perm_importance_last_fold.csv", index=False)

    # Plot top 15
    top = imp_df.head(15)
    fig, ax = plt.subplots()
    ax.barh(top["feature"], top["importance"])
    ax.invert_yaxis()
    ax.set_title(f"Permutation importance (test {ts}-{te})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "perm_importance_last_fold.png")
    plt.close(fig)

    print(f"\nPermutation importance saved to:")
    print(f"  - {PROCESSED / 'perm_importance_last_fold.csv'}")
    print(f"  - {FIG_DIR / 'perm_importance_last_fold.png'}")
else:
    print("\nSkipped permutation importance: last-fold test set is empty.")


# ---------------------------------------------------------------------
# Residuals plot on the last fold
# ---------------------------------------------------------------------
if mask_test.sum() > 0:
    y_true_last = y[mask_test]
    y_pred_last = pipe.predict(X[mask_test])
    fig, ax = plt.subplots()
    ax.scatter(y_true_last, y_pred_last)
    lims = [
        min(y_true_last.min(), y_pred_last.min()),
        max(y_true_last.max(), y_pred_last.max()),
    ]
    ax.plot(lims, lims)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Observed vs Predicted (test {ts}-{te})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "obs_vs_pred_last_fold.png")
    plt.close(fig)

    res = y_true_last - y_pred_last
    fig, ax = plt.subplots()
    ax.hist(res, bins=15)
    ax.set_title(f"Residuals (test {ts}-{te})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals_last_fold.png")
    plt.close(fig)

    print(f"\nLast-fold diagnostics saved to {FIG_DIR}")
