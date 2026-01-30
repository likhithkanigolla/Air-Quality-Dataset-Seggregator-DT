from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from src.data_loader import build_feature_table


def predict_for_station(model_path: Path, station_folder: Path, osm_csv: Path, locations_csv: Path = None):
    model = joblib.load(model_path)
    preproc = joblib.load(model_path.parent / "preprocessor.joblib")

    X, y = build_feature_table(station_folder, osm_csv, locations_csv=locations_csv)
    # use median of station features as representative input
    x_med = X.median(axis=0).to_frame().T
    x_p = preproc.transform(x_med)
    pred = model.predict(x_p)
    return float(pred[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model joblib file (model_rf.joblib etc.)")
    p.add_argument("--station-folder", required=True, help="Path to station folder (e.g. output/AP001)")
    p.add_argument("--osm-csv", required=True)
    p.add_argument("--locations-csv", default=None)
    args = p.parse_args()
    val = predict_for_station(Path(args.model), Path(args.station_folder), Path(args.osm_csv), Path(args.locations_csv) if args.locations_csv else None)
    print(val)


if __name__ == "__main__":
    main()
