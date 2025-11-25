import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/processed")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load core recommendation tables ----------
recs_path = DATA_DIR / "recommendations.csv"
recs_top_path = DATA_DIR / "recommendations_by_district_top3.csv"

recs = pd.read_csv(recs_path)
recs_top = pd.read_csv(recs_top_path) if recs_top_path.exists() else recs.copy()

# ---------- 1) Bar chart: composite score by species (per district) ----------
for district, g in recs.groupby("district"):
    g_plot = g.sort_values("composite_score", ascending=False)
    plt.figure(figsize=(7, 4))
    plt.bar(g_plot["item"], g_plot["composite_score"])
    plt.title(f"Composite score by species — {district}")
    plt.ylabel("Composite score (0–1)")
    plt.xlabel("Species")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = FIG_DIR / f"composite_by_species_{district.replace(' ', '_')}.png"
    plt.savefig(out, dpi=200)
    plt.close()

# ---------- 2) Scatter: productivity vs resilience ----------
plt.figure(figsize=(6, 5))
plt.scatter(recs["productivity_score"], recs["resilience_score"], s=200 * (recs["pred_mean_recent"] / recs["pred_mean_recent"].max()))
for _, r in recs.iterrows():
    plt.annotate(r["item"], (r["productivity_score"], r["resilience_score"]), xytext=(5, 5), textcoords="offset points")
plt.xlabel("Productivity score (0–1)")
plt.ylabel("Resilience score (0–1)")
plt.title("Productivity vs Resilience (size ~ predicted recent mean)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "scatter_productivity_vs_resilience.png", dpi=200)
plt.close()

# ---------- 3) Grouped bars: productivity and resilience by species (within district) ----------
for district, g in recs.groupby("district"):
    g_plot = g.sort_values("productivity_score", ascending=False)
    x = range(len(g_plot))
    width = 0.4
    plt.figure(figsize=(7, 4))
    plt.bar([i - width/2 for i in x], g_plot["productivity_score"], width=width, label="Productivity")
    plt.bar([i + width/2 for i in x], g_plot["resilience_score"],  width=width, label="Resilience")
    plt.xticks(list(x), g_plot["item"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score (0–1)")
    plt.title(f"Productivity vs Resilience — {district}")
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / f"prod_vs_resilience_{district.replace(' ', '_')}.png"
    plt.savefig(out, dpi=200)
    plt.close()

# ---------- 4) Feature importance (if available) ----------
fi_candidates = [
    DATA_DIR / "feature_importance_filtered.csv",
    DATA_DIR / "feature_importance.csv",
]
fi_path = None
for p in fi_candidates:
    if p.exists():
        fi_path = p
        break

if fi_path:
    fi = pd.read_csv(fi_path)
    # keep top 15 for readability
    fi_plot = fi.sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    plt.barh(fi_plot["feature"][::-1], fi_plot["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Feature importance ({fi_path.name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"feature_importance_{fi_path.stem}.png", dpi=200)
    plt.close()
