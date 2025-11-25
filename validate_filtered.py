# src/validate_filtered.py
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# -----------------------
# Paths & constants
# -----------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGDIR = PROCESSED / "figures"
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

INPUT = PROCESSED / "merged.csv"
CV_METRICS_OUT = PROCESSED / "cv_metrics_filtered.csv"
FOLD_PREDS_OUT = PROCESSED / "fold_predictions_filtered.csv"
PERM_IMP_CSV = PROCESSED / "perm_importance_filtered_last_fold.csv"
PERM_IMP_PNG = FIGDIR / "perm_importance_filtered_last_fold.png"
OBSPRED_PNG = FIGDIR / "obs_vs_pred_filtered_last_fold.png"

ELEMENT = "Livestock units per agricultural land area"
UNIT = "LSU/ha"

# Years for time-based CV: train <= (year-1), test == year
TEST_YEARS = [2013, 2014, 2015, 2016, 2017, 2018]

def load_filtered() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    # Enforce consistent dtypes
    df["year"] = df["year"].astype(int)
    # Filter to the chosen element/unit
    mask = (df["element"] == ELEMENT) & (df["unit"] == UNIT)
    df_f = df.loc[mask].copy()
    if df_f.empty:
        raise ValueError(
            "After filtering, no rows remain. "
            f"Check element=={ELEMENT!r} and unit=={UNIT!r} in {INPUT}"
        )
    return df_f

def make_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )
    pipe = Pipeline([("pre", pre), ("rf", model)])
    return pipe

def get_feature_names(pipe: Pipeline) -> list[str]:
    """
    Return the post-encoding feature names from the ColumnTransformer inside the pipeline.
    """
    pre: ColumnTransformer = pipe.named_steps["pre"]
    # sklearn >=1.0: get_feature_names_out
    try:
        names = pre.get_feature_names_out()
        # They come like "num__prectotcorr_ann", "cat__item_Cattle" etc.
        return [n.split("__", 1)[-1] for n in names]
    except Exception:
        # Fallback (rare)
        return [f"f_{i}" for i in range(pre.transform(pd.DataFrame()).shape[1])]

def main():
    df = load_filtered()

    # -----------------------
    # Features / target
    # -----------------------
    # Use climate features + simple categorical context
    num_feats = [c for c in df.columns if c.endswith("_ann") or c.endswith("_sum")]
    cat_feats = ["item", "district"]  # can include if available; harmless for one district
    target_col = "value"

    # Sanity checks
    for col in num_feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=num_feats + [target_col])

    # -----------------------
    # Time-based CV
    # -----------------------
    rows = []
    fold_preds = []

    for test_year in TEST_YEARS:
        train_mask = df["year"] <= (test_year - 1)
        test_mask = df["year"] == test_year

        X_train = df.loc[train_mask, num_feats + cat_feats]
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, num_feats + cat_feats]
        y_test = df.loc[test_mask, target_col].values

        if len(X_test) == 0 or len(X_train) == 0:
            # Skip folds where we don't have rows
            continue

        pipe = make_pipeline(num_feats, cat_feats)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        rows.append(
            {
                "train_end": test_year - 1,
                "test_start": test_year,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
            }
        )

        # store predictions for analysis
        tmp = pd.DataFrame(
            {
                "year": df.loc[test_mask, "year"].values,
                "item": df.loc[test_mask, "item"].values if "item" in df.columns else "NA",
                "district": df.loc[test_mask, "district"].values
                if "district" in df.columns
                else "NA",
                "y_true": y_test,
                "y_pred": y_pred,
                "fold_test_year": test_year,
            }
        )
        fold_preds.append(tmp)

        # If this is the LAST fold, produce diagnostics & permutation importance
        if test_year == TEST_YEARS[-1]:
            # Permutation importance on the last fold
            result = permutation_importance(
                pipe, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1
            )
            feat_names = get_feature_names(pipe)
            # Align lengths defensively
            n = min(len(feat_names), len(result.importances_mean))
            imp_df = pd.DataFrame(
                {
                    "feature": feat_names[:n],
                    "importance": result.importances_mean[:n],
                }
            ).sort_values("importance", ascending=False)
            imp_df.to_csv(PERM_IMP_CSV, index=False)

            # Bar plot
            top = imp_df.head(15)[::-1]
            plt.figure(figsize=(7, 6))
            plt.barh(top["feature"], top["importance"])
            plt.xlabel("Permutation importance (mean decrease in score)")
            plt.title("Permutation importance – filtered model (last fold)")
            plt.tight_layout()
            plt.savefig(PERM_IMP_PNG, dpi=150)
            plt.close()

            # Obs vs Pred scatter
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_pred, alpha=0.8)
            lims = [
                min(np.min(y_test), np.min(y_pred)),
                max(np.max(y_test), np.max(y_pred)),
            ]
            plt.plot(lims, lims, linestyle="--")
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.title(f"Observed vs Predicted – filtered model (test {test_year})")
            plt.tight_layout()
            plt.savefig(OBSPRED_PNG, dpi=150)
            plt.close()

    # -----------------------
    # Save CV metrics & preds
    # -----------------------
    if rows:
        cv_df = pd.DataFrame(rows)
        cv_df.to_csv(CV_METRICS_OUT, index=False)
        print("\nTime-based CV (filtered) across folds")
        print(cv_df)

        print("\nAverage over folds:")
        print(cv_df[["MAE", "RMSE", "R2"]].mean().to_frame("mean").T)

    if fold_preds:
        pd.concat(fold_preds, ignore_index=True).to_csv(FOLD_PREDS_OUT, index=False)

    # Report where files are
    print("\nSaved to:")
    print(f"  - {CV_METRICS_OUT}")
    print(f"  - {FOLD_PREDS_OUT}")
    print(f"  - {PERM_IMP_CSV}")
    print(f"  - {PERM_IMP_PNG}")
    print(f"  - {OBSPRED_PNG}")

if __name__ == "__main__":
    main()
