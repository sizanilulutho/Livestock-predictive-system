# src/merge_datasets.py (core idea)
from pathlib import Path
import pandas as pd

INTERIM = Path("data/interim")
PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)

def main():
    fao = pd.read_csv(INTERIM/"faostat_clean.csv")
    clim_ann = pd.read_csv(INTERIM/"climate_annual.csv")  # now includes prectotcorr_ann if available

    merged = pd.merge(
        fao, clim_ann, on="year", how="left"
    )
    out = PROC/"merged.csv"
    merged.to_csv(out, index=False)
    print(f" Merged → {out}  shape={merged.shape}")

if __name__ == "__main__":
    main()
