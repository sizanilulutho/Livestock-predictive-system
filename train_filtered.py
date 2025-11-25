# src/train_filtered.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("data/processed/merged.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------- Load --------------
df = pd.read_csv(DATA)

# Keep only needed columns up front
needed_cols = {"year","item","district","value",
               "element","unit",
               "prectotcorr_ann","prectotcorr_sum",
               "qv2m_ann","rh2m_ann","t2m_max_ann","t2m_range_ann"}
missing = needed_cols - set(df.columns)
if missing:
    raise ValueError(f"merged.csv missing columns: {missing}")

# -------------- Choose (element, unit) robustly --------------
# Normalize text for matching
df["_element_lc"] = df["element"].astype(str).str.strip().str.lower()
df["_unit_lc"]    = df["unit"].astype(str).str.strip().str.lower()

# Preferred combo: "livestock units per agricultural land" + "lsu/ha"
pref_element_kw = "livestock units per agricultural land"
pref_unit_exact = "lsu/ha"

pref_mask = df["_element_lc"].str.contains(pref_element_kw, regex=False) & (df["_unit_lc"] == pref_unit_exact)

if pref_mask.any():
    chosen = df.loc[pref_mask].copy()
else:
    # Fallback: pick the most common (element,unit) pair
    counts = (df.groupby(["_element_lc","_unit_lc"]).size()
                .sort_values(ascending=False))
    if counts.empty:
        raise ValueError("No (element, unit) pairs found at all.")
    top_el, top_un = counts.index[0]
    chosen = df[(df["_element_lc"]==top_el) & (df["_unit_lc"]==top_un)].copy()
    print(f"Note: preferred (element,unit) not found; using most frequent pair: element='{top_el}' | unit='{top_un}'")

if chosen.empty:
    raise ValueError("After filtering, no rows remain. Inspect 'element' and 'unit' in merged.csv.")

# -------------- Train/test split by year --------------
train = chosen[chosen["year"] <= 2017].copy()
test  = chosen[chosen["year"] >= 2018].copy()

if len(train)==0 or len(test)==0:
    raise ValueError(f"Train/test split empty. Train rows={len(train)} Test rows={len(test)}. "
                     f"Check your years in merged.csv.")

y_train = train["value"].astype(float)
y_test  = test["value"].astype(float)

# Climate-only numeric features (no leakage) + item/district as categorical
num_feats = ["prectotcorr_ann","prectotcorr_sum","qv2m_ann","rh2m_ann","t2m_max_ann","t2m_range_ann"]
cat_feats = ["item","district"]

X_train = train[num_feats + cat_feats].copy()
X_test  = test[num_feats + cat_feats].copy()

# -------------- Pipeline --------------
pre = ColumnTransformer([
    ("num", "passthrough", num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
])

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("pre", pre),
    ("rf", model)
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

# -------------- Metrics --------------
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print(f"Train years: <= 2017")
print(f"Test years: >= 2018")
print(f"Test MAE: {mae:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test R^2: {r2:.3f}")

# -------------- Feature importance (per climate + cats) --------------
# Pull back feature names from the preprocessor
ohe = pipe.named_steps["pre"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cat_feats))
feature_names = num_feats + cat_names

rf = pipe.named_steps["rf"]
importances = rf.feature_importances_
fi = (pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False))
fi.to_csv(OUT_DIR / "feature_importance_filtered.csv", index=False)

# -------------- Resilient species heuristic --------------
# As a simple signal, correlate value with precipitation on the test set per species
res_list = []
for item_name, g in test.groupby("item"):
    if g[num_feats].notna().all().all() and len(g) >= 4:
        # slope of value vs annual precip (simple linear fit)
        x = g["prectotcorr_ann"].astype(float).to_numpy()
        y = g["value"].astype(float).to_numpy()
        if np.isfinite(x).all() and np.isfinite(y).all():
            A = np.vstack([x, np.ones_like(x)]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            res_list.append({"item": item_name,
                             "slope_y_vs_precip": slope,
                             "abs_slope": abs(slope),
                             "n_test": len(g)})
res_df = (pd.DataFrame(res_list)
            .sort_values("abs_slope", ascending=True))  # smaller abs slope = more resilient
res_df.to_csv(OUT_DIR / "resilient_species_filtered.csv", index=False)

# -------------- Pasture ranking (recent) --------------
recent = chosen[chosen["year"] >= chosen["year"].max()-4]   # last ~5 years in data
recent_pred = pipe.predict(recent[num_feats + cat_feats])
rank_df = (recent.assign(pred=recent_pred)
                    .groupby(["item","district"], as_index=False)["pred"]
                    .mean()
                    .rename(columns={"pred":"pred_mean_recent"})
                    .sort_values("pred_mean_recent", ascending=False))
rank_df.to_csv(OUT_DIR / "pasture_ranking_filtered.csv", index=False)

# -------------- Save metrics summary --------------
with open(OUT_DIR / "model_metrics_filtered.txt", "w", encoding="utf-8") as f:
    f.write("Leakage-minimized climate model\n")
    f.write(f"Train years: <= 2017\n")
    f.write(f"Test years: >= 2018\n")
    f.write(f"Test MAE: {mae:.3f}\n")
    f.write(f"Test RMSE: {rmse:.3f}\n")
    f.write(f"Test R^2: {r2:.3f}\n")
    f.write("\nChosen (element, unit) pair:\n")
    f.write(str(chosen[['element','unit']].drop_duplicates().to_string(index=False)))
print("Saved:")
print("  - data/processed/model_metrics_filtered.txt")
print("  - data/processed/feature_importance_filtered.csv")
print("  - data/processed/resilient_species_filtered.csv")
print("  - data/processed/pasture_ranking_filtered.csv")
