from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_loader_chunks import build_station_table_from_folder, read_all_chunks, split_items
from datetime import datetime


MODEL_REGISTRY = {
    "linear": lambda: MultiOutputRegressor(LinearRegression()),
    "rf": lambda: MultiOutputRegressor(RandomForestRegressor(random_state=0, n_jobs=-1)),
    "gbr": lambda: MultiOutputRegressor(GradientBoostingRegressor(random_state=0)),
}


def train_multi(folder: str, out_dir: Path):
    # Read chunks first to detect Datetime presence and choose training mode
    df = read_all_chunks(Path(folder))
    # prefer row-level training if Datetime exists (use datetime-derived features)
    if "Datetime" in df.columns:
        # row-level training with datetime features
        target_candidates = ["PM2.5", "PM10", "AQI"]
        targets = [t for t in target_candidates if t in df.columns]
        if not targets:
            raise ValueError("No target columns found in chunk CSVs for row-level training")

        osm_cols = [c for c in df.columns if c not in ["StationId", "Datetime"] + targets]

        X_rows = []
        y_rows = []
        # ensure Datetime parsed
        df = df.copy()
        df["_dt_parsed"] = pd.to_datetime(df["Datetime"], errors="coerce")
        for _, r in df.iterrows():
            row = {"StationId": r.get("StationId")}
            # datetime-derived features
            dt = r.get("_dt_parsed")
            if pd.isna(dt):
                row["dt_hour"] = 0
                row["dt_dayofweek"] = 0
                row["dt_month"] = 0
                row["dt_is_weekend"] = 0
            else:
                row["dt_hour"] = int(dt.hour)
                row["dt_dayofweek"] = int(dt.dayofweek)
                row["dt_month"] = int(dt.month)
                row["dt_is_weekend"] = int(dt.weekday() >= 5)

            for c in osm_cols:
                row[f"osm_count__{c}"] = len(split_items(r.get(c, "")))

            X_rows.append(row)
            yrow = {t: r.get(t) for t in targets}
            y_rows.append(yrow)

        X = pd.DataFrame(X_rows)
        y = pd.DataFrame(y_rows)
        mask = ~y.isna().any(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
    else:
        # fall back to station-level aggregation
        X, y = build_station_table_from_folder(folder)
        # drop stations with NaN targets
        y = y.dropna(how="any")
        X = X.loc[y.index]

    # keep StationId as categorical feature
    X = X.reset_index(drop=True)
    cat_cols = ["StationId"] if "StationId" in X.columns else []
    num_cols = [c for c in X.columns if c not in cat_cols]

    # OneHotEncoder API changed between sklearn versions: use the available kwarg
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preproc = ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", StandardScaler(), num_cols)
    ], remainder="drop")

    X_p = preproc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_p, y.values, test_size=0.2, random_state=0)

    results = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in MODEL_REGISTRY.items():
        model = factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds, multioutput="variance_weighted"))
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2}
        joblib.dump(model, out_dir / f"model_{name}_multi.joblib")

    joblib.dump(preproc, out_dir / "preprocessor_multi.joblib")
    with open(out_dir / "metrics_multi.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station-folder", required=True)
    p.add_argument("--out-dir", default="models_multi")
    args = p.parse_args()
    out = Path(args.out_dir)
    results = train_multi(args.station_folder, out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
