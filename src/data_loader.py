import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple


COMMON_POLLUTANT_CANDIDATES = [
    "PM2.5",
    "pm2.5",
    "pm25",
    "PM25",
    "pm2_5",
    "pm10",
    "PM10",
    "aqi",
    "value",
]


def read_ap_station_folder(folder: Path) -> pd.DataFrame:
    """Read all CSV chunks in a station output folder and return concatenated DataFrame."""
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            # fallback: try with low_memory
            dfs.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(dfs, ignore_index=True)
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """Pick a target column name from common pollutant names present in the DataFrame."""
    cols = set(df.columns.str.lower())
    for cand in COMMON_POLLUTANT_CANDIDATES:
        if cand.lower() in cols:
            # return original-case column name
            for c in df.columns:
                if c.lower() == cand.lower():
                    return c
    # fallback: choose first numeric column that is not lat/lon/time
    for c in df.columns:
        if c.lower() in ("latitude", "longitude", "lat", "lon", "time", "timestamp"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No suitable numeric target column found in station data")


def summarize_osm_row(osm_row: pd.Series, cols_to_count=None) -> dict:
    if cols_to_count is None:
        cols_to_count = [
            "amenity",
            "highway",
            "shop",
            "leisure",
            "building",
            "natural",
            "_unique_feature_types",
        ]
    out = {}
    for c in cols_to_count:
        if c in osm_row.index:
            val = osm_row.get(c, "")
            if pd.isna(val) or val == "":
                out[f"osm_count_{c}"] = 0
            else:
                # assume comma separated
                if isinstance(val, (int, float)):
                    out[f"osm_count_{c}"] = int(val)
                else:
                    out[f"osm_count_{c}"] = len([s for s in str(val).split(",") if s.strip()])
        else:
            out[f"osm_count_{c}"] = 0
    return out


def build_feature_table(
    station_folder: Path,
    osm_csv: Path,
    locations_csv: Path = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load time-series for the station, extract a target and add numeric OSM features.

    Returns X (DataFrame of features) and y (Series target).
    """
    station_id = station_folder.name
    df = read_ap_station_folder(station_folder)

    target_col = detect_target_column(df)
    # keep only numeric columns + target
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])] 
    if target_col not in numeric_cols:
        # coerce target to numeric if needed
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        numeric_cols.append(target_col)

    base = df[numeric_cols].copy()

    # load osm
    osm_df = pd.read_csv(osm_csv)
    osm_row = osm_df[osm_df["station_id"].astype(str) == station_id]
    if osm_row.shape[0] == 0:
        # try match original_station_id or without prefix
        osm_row = osm_df[osm_df["original_station_id"].astype(str).str.contains(station_id, na=False)]
    if osm_row.shape[0] == 0:
        # no OSM features found, create zeros
        osm_features = {"osm_count_amenity": 0, "osm_count_highway": 0, "osm_count_shop": 0, "osm_count_leisure": 0, "osm_count_building": 0, "osm_count_natural": 0, "osm_count__unique_feature_types": 0}
    else:
        osm_features = summarize_osm_row(osm_row.iloc[0])

    # create repeated osm features for each row in base
    for k, v in osm_features.items():
        base[k] = v

    # optional: add station lat/lon from locations_csv
    if locations_csv is not None and Path(locations_csv).exists():
        locs = pd.read_csv(locations_csv)
        match = locs[locs["station_id"].astype(str).str.contains(station_id, na=False)]
        if match.shape[0] > 0:
            lat = match.iloc[0].get("latitude", np.nan)
            lon = match.iloc[0].get("longitude", np.nan)
            base["station_lat"] = float(lat)
            base["station_lon"] = float(lon)
        else:
            base["station_lat"] = np.nan
            base["station_lon"] = np.nan

    X = base.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0)
    y = base[target_col].astype(float).fillna(0)
    return X, y


if __name__ == "__main__":
    # quick manual test helper
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("station_folder")
    p.add_argument("osm_csv")
    p.add_argument("--locations_csv", default=None)
    args = p.parse_args()
    X, y = build_feature_table(Path(args.station_folder), Path(args.osm_csv), Path(args.locations_csv) if args.locations_csv else None)
    print("X.shape", X.shape)
    print("y.shape", y.shape)
