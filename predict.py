import argparse
import sys
import pathlib
import pandas as pd
import joblib

EXPECTED_MODEL = pathlib.Path("data/processed/model_filtered.joblib")

def parse_args():
    p = argparse.ArgumentParser(
        description="Load the saved LSU/ha model and predict for new data."
    )
    p.add_argument("--model", type=str, default=str(EXPECTED_MODEL),
                   help="Path to the saved joblib model (default: data/processed/model_filtered.joblib)")
    p.add_argument("--input", type=str, required=True,
                   help="Path to input CSV with columns: year,item,district and climate features")
    p.add_argument("--output", type=str, default=None,
                   help="Where to write predictions CSV (default: alongside input with _predictions suffix)")
    p.add_argument("--filter_year_min", type=int, default=None,
                   help="If set, keep only rows with year >= this value (e.g., 2018 or 2023)")
    return p.parse_args()

def main():
    args = parse_args()
    model_path = pathlib.Path(args.model)
    input_path = pathlib.Path(args.input)

    if not model_path.exists():
        sys.exit(f"ERROR: Model file not found: {model_path}\n"
                 f"Train/save it first (model_filtered.joblib).")

    if not input_path.exists():
        sys.exit(f"ERROR: Input CSV not found: {input_path}")

    # Load trained pipeline and feature schema
    bundle = joblib.load(model_path)
    pipe = bundle["pipe"]
    num_feats = bundle["num_feats"]
    cat_feats = bundle["cat_feats"]
    required_cols = num_feats + cat_feats

    # Read data
    df = pd.read_csv(input_path)

    # Optional year filter
    if args.filter_year_min is not None:
        if "year" not in df.columns:
            sys.exit("ERROR: --filter_year_min was provided but 'year' column is missing in input CSV.")
        df = df[df["year"] >= args.filter_year_min].copy()

    if df.empty:
        sys.exit("ERROR: No rows to predict after filtering.")

    # Check required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit("ERROR: Input CSV missing required columns:\n"
                 f"  {missing}\n\n"
                 f"Expected at least these columns:\n  {required_cols}\n"
                 "Tip: Use merge_datasets.py output or build a CSV with the same schema.")

    # Predict
    X = df[required_cols].copy()
    preds = pipe.predict(X)

    out = df.copy()
    out["prediction"] = preds  # predicted LSU/ha

    # Default output path
    if args.output is None:
        out_path = input_path.with_name(input_path.stem + "_predictions.csv")
    else:
        out_path = pathlib.Path(args.output)

    out.to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")
    print("Preview:")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()