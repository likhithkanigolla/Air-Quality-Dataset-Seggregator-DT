from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List


def read_all_chunks(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(dfs, ignore_index=True)


def split_items(cell: object) -> List[str]:
    if pd.isna(cell):
        return []
    if isinstance(cell, (int, float)):
        return []
    return [s.strip() for s in str(cell).split(",") if s.strip()]


def station_level_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate chunk rows to station-level features and targets.

    Returns X (DataFrame indexed by StationId) and y (DataFrame with targets PM2.5, PM10, AQI).
    """
    # ensure StationId exists
    if "StationId" not in df.columns and "stationid" in df.columns:
        df = df.rename(columns={"stationid": "StationId"})

    if "StationId" not in df.columns:
        raise ValueError("StationId column not found in chunks")

    # identify targets available
    target_candidates = ["PM2.5", "PM10", "AQI"]
    targets = [t for t in target_candidates if t in df.columns]

    # identify osm-like columns: columns after AQI_Bucket or common names
    osm_cols = [
        c for c in df.columns
        if c.lower() not in ["stationid", "datetime"] + [t.lower() for t in targets] + ["aqi_bucket"]
    ]

    # For each station, compute:
    # - target median per station
    # - for each osm_col, number of unique items across rows
    groups = df.groupby("StationId")
    X_rows = []
    y_rows = []
    for station, g in groups:
        row = {"StationId": station}
        for c in osm_cols:
            # gather unique items across rows
            items = set()
            for cell in g[c].fillna(""):
                for it in split_items(cell):
                    items.add(it)
            row[f"osm_unique_count__{c}"] = len(items)
        # also add average counts per row
        for c in osm_cols:
            counts = [len(split_items(cell)) for cell in g[c].fillna("")]
            row[f"osm_avg_count__{c}"] = float(np.mean(counts)) if counts else 0.0

        X_rows.append(row)

        yrow = {"StationId": station}
        for t in targets:
            yrow[t] = g[t].median(skipna=True)
        y_rows.append(yrow)

    X = pd.DataFrame(X_rows).set_index("StationId")
    y = pd.DataFrame(y_rows).set_index("StationId")
    # drop columns that are constant zero
    X = X.loc[:, (X != 0).any(axis=0)]
    return X, y


def build_station_table_from_folder(folder: str):
    df = read_all_chunks(Path(folder))
    X, y = station_level_features(df)
    return X, y


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    args = p.parse_args()
    X, y = build_station_table_from_folder(args.folder)
    print("Stations:", len(X))
    print("X columns:", list(X.columns)[:20])
    print("Targets:", list(y.columns))
