from pathlib import Path
import argparse
import json
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from src.data_loader import build_feature_table


MODEL_REGISTRY = {
    "linear": LinearRegression,
    "rf": RandomForestRegressor,
    "gbr": GradientBoostingRegressor,
}


def train_models(X, y, out_dir: Path, random_state: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    X = X.copy()
    y = y.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    preproc = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    X_train_p = preproc.fit_transform(X_train)
    X_test_p = preproc.transform(X_test)

    results = {}
    # train each model
    for name, ModelCls in MODEL_REGISTRY.items():
        if name == "linear":
            model = ModelCls()
        else:
            model = ModelCls(random_state=random_state)
        model.fit(X_train_p, y_train)
        preds = model.predict(X_test_p)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2}

        # save model
        joblib.dump(model, out_dir / f"model_{name}.joblib")

    # save preprocessor and metrics
    joblib.dump(preproc, out_dir / "preprocessor.joblib")
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station-folder", required=True, help="Path to station folder (e.g. output/AP001)")
    p.add_argument("--osm-csv", required=True, help="Path to station_osm_features_filtered_110.csv")
    p.add_argument("--locations-csv", default=None, help="Path to station_locations.csv (optional)")
    p.add_argument("--out-dir", default="models", help="Directory to save trained models")
    p.add_argument("--max-samples", type=int, default=0, help="If >0, sample this many rows for quick tests")
    args = p.parse_args()

    station_folder = Path(args.station_folder)
    osm_csv = Path(args.osm_csv)
    loc_csv = Path(args.locations_csv) if args.locations_csv else None
    out_dir = Path(args.out_dir)

    X, y = build_feature_table(station_folder, osm_csv, locations_csv=loc_csv)
    if args.max_samples and args.max_samples > 0:
        sample_idx = np.random.RandomState(0).choice(len(X), size=min(args.max_samples, len(X)), replace=False)
        X = X.iloc[sample_idx]
        y = y.iloc[sample_idx]

    if len(X) < 10:
        print(f"Warning: very small training set ({len(X)} rows)")

    results = train_models(X, y, out_dir)
    print("Training results:\n", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
