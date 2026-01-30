import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from typing import List


def load_osm_input(osm_json_arg: str):
    p = Path(osm_json_arg)
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    # try parse as JSON string
    try:
        return json.loads(osm_json_arg)
    except Exception:
        raise ValueError("`--osm-json` must be a path to a JSON file or a JSON string")


def build_row_from_input(preproc, station_id: str, osm_input: dict):
    # get transformer column lists
    transformers = {t[0]: t for t in preproc.transformers_}
    # cat transformer at key 'cat' typically
    cat_name, cat_trans, cat_cols = preproc.transformers_[0]
    num_name, num_trans, num_cols = preproc.transformers_[1]

    row = {}
    # StationId
    if cat_cols:
        row[cat_cols[0]] = station_id

    # for each numeric column (likely names like osm_count__<col> or osm_avg_count__<col>)
    for nm in num_cols:
        val = 0
        # datetime-derived features expected names: dt_hour, dt_dayofweek, dt_month, dt_is_weekend
        if nm in ("dt_hour", "dt_dayofweek", "dt_month", "dt_is_weekend"):
            dt_val = osm_input.get("datetime") or osm_input.get("Datetime")
            if not dt_val:
                val = 0
            else:
                try:
                    parsed = pd.to_datetime(dt_val)
                    if nm == "dt_hour":
                        val = int(parsed.hour)
                    elif nm == "dt_dayofweek":
                        val = int(parsed.dayofweek)
                    elif nm == "dt_month":
                        val = int(parsed.month)
                    elif nm == "dt_is_weekend":
                        val = int(parsed.weekday() >= 5)
                except Exception:
                    val = 0
            row[nm] = val
            continue
        if nm.startswith("osm_count__"):
            raw = nm.replace("osm_count__", "")
            if raw in osm_input:
                v = osm_input[raw]
                if isinstance(v, list):
                    val = len(v)
                elif isinstance(v, str):
                    val = len([s for s in v.split(",") if s.strip()])
                elif isinstance(v, (int, float)):
                    val = float(v)
        elif nm.startswith("osm_avg_count__"):
            raw = nm.replace("osm_avg_count__", "")
            # user may provide avg or list
            if raw in osm_input:
                v = osm_input[raw]
                if isinstance(v, (int, float)):
                    val = float(v)
                elif isinstance(v, list):
                    val = float(len(v))
                elif isinstance(v, str):
                    val = float(len([s for s in v.split(",") if s.strip()]))
        else:
            # if user provided a value matching exact numeric column
            if nm in osm_input:
                val = osm_input[nm]
        row[nm] = val

    return pd.DataFrame([row])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="models_multi/AP001", help="Directory containing model_*.joblib files")
    p.add_argument("--preproc", default="models_multi/AP001/preprocessor_multi.joblib")
    p.add_argument("--station-id", required=True)
    p.add_argument("--osm-json", required=True, help="Path to JSON or JSON string with OSM features. Keys should be OSM column names (e.g., 'amenity','highway') or counts named like 'amenity' -> list or comma string")
    args = p.parse_args()

    preproc = joblib.load(args.preproc)
    osm_input = load_osm_input(args.osm_json)

    # build input row and transform once
    row = build_row_from_input(preproc, args.station_id, osm_input)
    X_p = preproc.transform(row)

    # find model files
    models_path = Path(args.models_dir)
    model_files: List[Path] = []
    if models_path.exists() and models_path.is_dir():
        # match common naming used in training: model_<name>_multi.joblib
        model_files = sorted(models_path.glob("model_*_multi.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {args.models_dir}")

    out_names = ["PM2.5", "PM10", "AQI"]
    results = {}
    for mf in model_files:
        model = joblib.load(mf)
        preds = model.predict(X_p)
        # derive model short name
        stem = mf.stem  # e.g., model_rf_multi
        short = stem.replace("model_", "").replace("_multi", "")
        results[short] = {out_names[i]: float(preds[0][i]) for i in range(preds.shape[1])}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
