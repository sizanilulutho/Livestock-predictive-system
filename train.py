# src/train.py
import pathlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

DATA = pathlib.Path("data/processed/merged.csv")
OUT_DIR = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_END_YEAR = 2017  # train <= 2017, test >= 2018

# 1) Load
df = pd.read_csv(DATA)

# 2) Choose target and features
target = "value"

# Keep climate features you have (from your column list)
num_feats = [
    "prectotcorr_ann",
    "prectotcorr_sum",
    "qv2m_ann",
    "rh2m_ann",
    "t2m_max_ann",
    "t2m_range_ann",
]

# Add a couple of categorical descriptors (if present)
cat_feats = []
for c in ["item", "district", "unit", "element"]:
    if c in df.columns:
        cat_feats.append(c)

keep_cols = ["year", target] + num_feats + cat_feats
df = df[[c for c in keep_cols if c in df.columns]].copy()

# 3) Basic cleaning
df = df.dropna(subset=[target] + num_feats)  # ensure target & numerics present
df = df[df["year"].notna()]
df["year"] = df["year"].astype(int)

# 4) Split by time
train = df[df["year"] <= TRAIN_END_YEAR].copy()
test  = df[df["year"] >= TRAIN_END_YEAR + 1].copy()

if len(train) == 0 or len(test) == 0:
    raise ValueError("Not enough data after time split. Check 'year' coverage.")

X_train = train.drop(columns=[target])
y_train = train[target].values

X_test = test.drop(columns=[target])
y_test = test[target].values

# 5) Build pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", [f for f in num_feats if f in X_train.columns]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cat_feats if c in X_train.columns]),
    ],
    remainder="drop",
)

model = RandomForestRegressor(
    n_estimators=400,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    min_samples_leaf=2,
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

# 6) Train
pipe.fit(X_train, y_train)

# 7) Evaluate
y_pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
r2 = r2_score(y_test, y_pred)

with open(OUT_DIR / "model_metrics.txt", "w") as f:
    f.write(f"Train years: <= {TRAIN_END_YEAR}\n")
    f.write(f"Test years: >= {TRAIN_END_YEAR+1}\n")
    f.write(f"Test MAE: {mae:.3f}\n")
    f.write(f"Test RMSE: {rmse:.3f}\n")
    f.write(f"Test R2: {r2:.3f}\n")

print(f"Test MAE: {mae:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test R^2: {r2:.3f}")

# 8) Feature importance (approximate, via model’s importances mapped to columns)
#    Get the post-encoding feature names:
def get_feature_names(ct: ColumnTransformer, X_cols: list[str]) -> list[str]:
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            # e.g. OneHotEncoder
            fn = trans.get_feature_names_out(cols)
            names.extend(fn.tolist())
        elif trans == "passthrough":
            # cols are numeric raw columns
            if isinstance(cols, slice):
                cols = X_cols[cols]
            names.extend(list(cols))
        else:
            # other transformers
            if isinstance(cols, slice):
                cols = X_cols[cols]
            names.extend(list(cols))
    return names

X_cols = X_train.columns.tolist()
feat_names = get_feature_names(pipe.named_steps["prep"], X_cols)
importances = pipe.named_steps["model"].feature_importances_
fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)

# 9) Resilience proxy per species (item):
#    For each item present in the TEST set, regress predicted value on precip (prectotcorr_ann).
#    Smaller absolute slope means less sensitivity to rainfall → more resilient.
res_rows = []
if "item" in X_test.columns and "prectotcorr_ann" in X_test.columns:
    tmp = X_test.copy()
    tmp["y_pred"] = y_pred
    tmp[target] = y_test
    for it, g in tmp.groupby("item"):
        if g["prectotcorr_ann"].notna().sum() >= 10:
            X_lr = g[["prectotcorr_ann"]].values
            y_lr = g["y_pred"].values
            lr = LinearRegression()
            lr.fit(X_lr, y_lr)
            slope = float(lr.coef_[0])
            res_rows.append({"item": it, "slope_y_vs_precip": slope, "abs_slope": abs(slope), "n_test": len(g)})
    if res_rows:
        resilience = pd.DataFrame(res_rows).sort_values("abs_slope", ascending=True)
        resilience.to_csv(OUT_DIR / "resilient_species.csv", index=False)

# 10) Pasture/grazing suitability map (by district & item; recent climate years, ≥2018):
#     Average predicted value per district & item using test period.
pasture_rows = []
if "district" in X_test.columns and "item" in X_test.columns:
    tmp = X_test.copy()
    tmp["y_pred"] = y_pred
    grp = tmp.groupby(["item", "district"], as_index=False)["y_pred"].mean().rename(columns={"y_pred":"pred_mean_recent"})
    grp = grp.sort_values(["item", "pred_mean_recent"], ascending=[True, False])
    grp.to_csv(OUT_DIR / "pasture_ranking.csv", index=False)

print("Saved:")
print("  - data/processed/model_metrics.txt")
print("  - data/processed/feature_importance.csv")
if (OUT_DIR / "resilient_species.csv").exists():
    print("  - data/processed/resilient_species.csv")
if (OUT_DIR / "pasture_ranking.csv").exists():
    print("  - data/processed/pasture_ranking.csv")
