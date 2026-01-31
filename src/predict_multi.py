from pathlib import Path
import argparse
import joblib
import pandas as pd
from src.data_loader_chunks import read_all_chunks, split_items


def predict_for_folder(model_path: Path, folder: Path):
    # load preprocessor and model
    model = joblib.load(model_path)
    preproc = joblib.load(model_path.parent / "preprocessor_multi.joblib")

    # Build row-level features the same way training fallback did: StationId + osm_count__<col>
    df = read_all_chunks(folder)
    # identify target-like cols to exclude
    target_candidates = ["PM2.5", "PM10", "AQI"]
    targets = [t for t in target_candidates if t in df.columns]
    osm_cols = [c for c in df.columns if c not in ["StationId", "Datetime"] + targets]

    X_rows = []
    for _, r in df.iterrows():
        row = {"StationId": r.get("StationId")}
        for c in osm_cols:
            row[f"osm_count__{c}"] = len(split_items(r.get(c, "")))
        X_rows.append(row)

    X = pd.DataFrame(X_rows)
    # transform and predict; aggregate predictions per StationId (median)
    X_reset = X.reset_index(drop=True)
    X_p = preproc.transform(X_reset)
    preds = model.predict(X_p)
    # model predicts multi-output per row; aggregate per station
    pred_df = pd.DataFrame(preds, columns=[f"out_{i}" for i in range(preds.shape[1])])
    pred_df["StationId"] = X_reset["StationId"].values
    agg = pred_df.groupby("StationId").median()
    # rename columns if possible
    # assume outputs correspond to PM2.5, PM10, AQI order used in training
    expected = ["PM2.5", "PM10", "AQI"]
    cols = list(agg.columns)
    rename_map = {cols[i]: expected[i] for i in range(min(len(cols), len(expected)))}
    agg = agg.rename(columns=rename_map)
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--station-folder", required=True)
    args = p.parse_args()
    preds = predict_for_folder(Path(args.model), Path(args.station_folder))
    print(preds)


if __name__ == "__main__":
    main()
