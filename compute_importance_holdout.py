# src/compute_importance_holdout.py
"""
Compute permutation importance on the true hold-out (year >= 2018).
- Filters to LSU/ha rows if a 'unit' column exists (unit contains 'LSU', case-insensitive).
- Target defaults to 'value' (override with --target NAME).
- Uses the trained bundle at data/processed/model_filtered.joblib.
Outputs:
  data/processed/figures/permutation_importance_holdout.csv
  data/processed/figures/permutation_importance_holdout.png
Run:
  python src/compute_importance_holdout.py
  # or to force target name:
  python src/compute_importance_holdout.py --target value
"""
import argparse
import pathlib
import sys
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

MODEL_PATH = pathlib.Path("data/processed/model_filtered.joblib")
DATA_PATH  = pathlib.Path("data/processed/merged.csv")
OUT_DIR    = pathlib.Path("data/processed/figures"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_bundle(path: pathlib.Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    bundle = joblib.load(path)
    for k in ("pipe", "num_feats", "cat_feats"):
        if k not in bundle:
            raise KeyError(f"Missing '{k}' in model bundle: {list(bundle.keys())}")
    return bundle["pipe"], list(bundle["num_feats"]), list(bundle["cat_feats"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="value", help="Target column name (default: value)")
    ap.add_argument("--unit_contains", type=str, default="LSU", help="Keep rows where unit contains this text (case-insensitive). Default: LSU")
    args = ap.parse_args()

    if not DATA_PATH.exists():
        sys.exit(f"Data file not found: {DATA_PATH}")

    pipe, num_feats, cat_feats = load_bundle(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    # Optional filtering by unit (keep LSU/ha)
    if "unit" in df.columns:
        before = len(df)
        df = df[df["unit"].astype(str).str.contains(args.unit_contains, case=False, na=False)].copy()
        after = len(df)
        if after == 0:
            sys.exit(f"No rows left after unit filter: unit contains '{args.unit_contains}'. "
                     f"Check your data/processed/merged.csv")
        print(f"Filtered by unit contains '{args.unit_contains}': {before} -> {after} rows.")

    # Require 'year' to define hold-out
    if "year" not in df.columns:
        raise ValueError("Expected a 'year' column to define the hold-out (>=2018).")

    # Ensure target exists
    target_col = args.target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    # Define features as used by training
    feature_cols = list(num_feats) + list(cat_feats)
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing expected feature columns in data: {missing_feats}")

    # Hold-out split (align with your training setup)
    test = df[df["year"] >= 2018].copy()
    if test.empty:
        raise ValueError("Hold-out set is empty (no rows with year >= 2018).")

    X_test = test[feature_cols]
    y_test = test[target_col]

    # Score and permutation importance on the hold-out only
    r2 = pipe.score(X_test, y_test)

    perm = permutation_importance(
        pipe, X_test, y_test,
        scoring="r2", n_repeats=25, random_state=42, n_jobs=-1
    )
    imp = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = OUT_DIR / "permutation_importance_holdout.csv"
    imp.to_csv(csv_path, index=False)

    print(f"\nHold-out R^2: {r2:.3f}")
    print("\nTop features on hold-out (ΔR² mean ± std):")
    top = imp.head(12).copy()
    top["importance_mean"] = top["importance_mean"].round(6)
    top["importance_std"] = top["importance_std"].round(6)
    print(top.to_string(index=False))
    print("\nSaved CSV:", csv_path)

    # Optional bar chart
    try:
        import matplotlib.pyplot as plt
        topk = imp.head(20).iloc[::-1]
        plt.figure(figsize=(9, 6))
        plt.barh(topk["feature"], topk["importance_mean"])
        plt.xlabel("Permutation importance (Δ R²)")
        plt.tight_layout()
        png_path = OUT_DIR / "permutation_importance_holdout.png"
        plt.savefig(png_path, dpi=220)
        print("Saved PNG:", png_path)
    except Exception as e:
        print("Plotting skipped:", e)

if __name__ == "__main__":
    main()
