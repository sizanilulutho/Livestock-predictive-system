from pathlib import Path
import pandas as pd

def _find_parameter_header_row(csv_path: Path) -> int:
    """Return the line index where the header starts with 'PARAMETER,'."""
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().upper().startswith("PARAMETER,"):
                return i
    raise ValueError("POWER: couldn't find a line starting with 'PARAMETER,'.")

def _read_power_parameter_wide(csv_path: Path) -> pd.DataFrame:
    """
    Reads POWER file like:
      -BEGIN HEADER-
      ...
      PARAMETER,YEAR,LAT,LON,JAN,...,DEC,ANN
      ALLSKY_SFC_SW_DWN,2000,-22.5,16.5,29.29,...,23.78
      ...

    Returns tidy monthly df with columns:
      year, month, date, allsky_sfc_sw_dwn
    (values averaged across LAT/LON grid cells)
    """
    hdr = _find_parameter_header_row(csv_path)
    # read starting at header row
    df = pd.read_csv(csv_path, skiprows=hdr)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"parameter","year","lat","lon","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"POWER: unexpected columns. Found: {list(df.columns)[:15]}")

    # keep only the parameter(s) you care about; add more if needed
    keep_params = {"allsky_sfc_sw_dwn"}
    df = df[df["parameter"].str.lower().isin(keep_params)].copy()

    # melt monthly columns to long format
    month_cols = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    long = df.melt(
        id_vars=["parameter","year","lat","lon"],
        value_vars=month_cols,
        var_name="month_name",
        value_name="value"
    )

    # map month name -> number
    month_map = {m:i+1 for i,m in enumerate(month_cols)}
    long["month"] = long["month_name"].str.lower().map(month_map)
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # average across the grid cells (lat/lon) for each year-month
    grp = (
        long.groupby(["parameter","year","month"], as_index=False)["value"]
            .mean()
            .rename(columns={"value":"mean_value"})
    )

    # pivot so each parameter becomes its own column (here: allsky_sfc_sw_dwn)
    wide = grp.pivot(index=["year","month"], columns="parameter", values="mean_value").reset_index()
    wide.columns = [c if isinstance(c,str) else c for c in wide.columns]
    wide.columns = [c.lower().strip().replace(" ","_") for c in wide.columns]

    # build date column (1st of month)
    wide["date"] = pd.to_datetime(dict(year=wide["year"], month=wide["month"], day=1), errors="coerce")
    # sort
    wide = wide.sort_values(["year","month"]).reset_index(drop=True)

    # ensure canonical column name for your merge script
    wide = wide.rename(columns={"allsky_sfc_sw_dwn":"allsky_sfc_sw_dwn"})
    return wide[["year","month","date","allsky_sfc_sw_dwn"]]

def clean_climate():
    raw = Path("data/raw/POWER_Regional_Monthly_2000_2022.csv")
    if not raw.exists():
        alt = Path("data/raw/Power_climate.csv")
        raw = alt if alt.exists() else raw
    if not raw.exists():
        raise FileNotFoundError(
            f"Missing POWER file at {raw} (or data/raw/Power_climate.csv)."
        )

    out_dir = Path("data/interim"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "climate_clean.csv"

    df = _read_power_parameter_wide(raw)
    df.to_csv(out_path, index=False)
    print(f" Saved {out_path} (rows={len(df)}) with columns {list(df.columns)}")
