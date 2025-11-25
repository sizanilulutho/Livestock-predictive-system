# src/make_recommendations.py
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data/processed")
# Prefer filtered outputs; fall back to unfiltered if needed
resilient_paths = [DATA / "resilient_species_filtered.csv", DATA / "resilient_species.csv"]
pasture_paths   = [DATA / "pasture_ranking_filtered.csv", DATA / "pasture_ranking.csv"]

def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {paths}")

def _safe_minmax_norm(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(1.0, index=s.index)  # no variation → treat all as 1
    return (s - s.min()) / (s.max() - s.min())

def main():
    resilient_file = _first_existing(resilient_paths)
    pasture_file   = _first_existing(pasture_paths)

    res = pd.read_csv(resilient_file)
    pas = pd.read_csv(pasture_file)

    # Expect columns:
    #   res: item, slope_y_vs_precip, abs_slope, n_test
    #   pas: item, district, pred_mean_recent
    need_res = {"item", "abs_slope"}
    need_pas = {"item", "district", "pred_mean_recent"}
    if not need_res.issubset(res.columns):
        raise ValueError(f"{resilient_file} missing {need_res - set(res.columns)}")
    if not need_pas.issubset(pas.columns):
        raise ValueError(f"{pasture_file} missing {need_pas - set(pas.columns)}")

    # Merge on item (species), keep district from pasture
    df = pas.merge(res[["item", "abs_slope"]], on="item", how="left")

    # Scores:
    # - Productivity: higher pred_mean_recent is better → normalize 0..1 directly
    # - Resilience: smaller abs_slope is better → invert min-max so lower slope → higher score
    df["productivity_score"] = _safe_minmax_norm(df["pred_mean_recent"])
    # normalize slope to 0..1, then invert
    slope_norm = _safe_minmax_norm(df["abs_slope"])
    df["resilience_score"] = 1.0 - slope_norm

    # Composite score (tune weights if you like)
    W_PROD = 0.60
    W_RES  = 0.40
    df["composite_score"] = W_PROD * df["productivity_score"] + W_RES * df["resilience_score"]

    # Ranks
    df = df.sort_values(["composite_score", "productivity_score"], ascending=False, kind="mergesort")
    df["global_rank"] = np.arange(1, len(df) + 1)
    df["rank_within_district"] = df.groupby("district")["composite_score"] \
                                    .rank(method="first", ascending=False).astype(int)

    # Select and order columns
    out_cols = [
        "district", "item",
        "pred_mean_recent", "abs_slope",
        "productivity_score", "resilience_score", "composite_score",
        "rank_within_district", "global_rank"
    ]
    df_out = df[out_cols].copy()

    # Save full recommendations
    OUT1 = DATA / "recommendations.csv"
    df_out.to_csv(OUT1, index=False)

    # Top N per district (e.g., 3)
    TOP_N = 3
    top_by_district = (df_out.sort_values(["district", "rank_within_district"])
                             .groupby("district", as_index=False)
                             .head(TOP_N))
    OUT2 = DATA / "recommendations_by_district_top3.csv"
    top_by_district.to_csv(OUT2, index=False)

    # Simple text summary
    lines = []
    lines.append("Recommendations summary")
    lines.append(f"Source: {resilient_file.name} + {pasture_file.name}")
    lines.append(f"Weighting: productivity={W_PROD:.2f}, resilience={W_RES:.2f}")
    lines.append("")
    lines.append("Top 3 per district:")
    for d, grp in top_by_district.groupby("district"):
        lines.append(f"- {d}: " + ", ".join(
            f"{r.item} (score={r.composite_score:.3f})" for _, r in grp.iterrows()
        ))
    OUT3 = DATA / "recommendations_summary.txt"
    OUT3.write_text("\n".join(lines), encoding="utf-8")

    # Console preview
    print(f"Saved:\n  - {OUT1}\n  - {OUT2}\n  - {OUT3}")
    print("\nPreview (top 10 overall):")
    print(df_out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
