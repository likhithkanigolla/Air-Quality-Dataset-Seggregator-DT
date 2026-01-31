from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
import sys


def load_osm_input(osm_input: str) -> dict:
    """Load OSM features mapping from a JSON string or a filepath."""
    p = Path(osm_input)
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    else:
        # try parse as JSON string
        try:
            return json.loads(osm_input)
        except Exception:
            print("Failed to parse --osm-json as file or JSON string", file=sys.stderr)
            raise


def build_input_row(preproc, station_id: str, osm_map: dict) -> pd.DataFrame:
    """Construct a single-row DataFrame with columns expected by preprocessor.

    Strategy:
    - Inspect preproc.transformers_ to get categorical and numeric column names used at fit time.
    - Fill numeric columns with zeros, then populate with counts from `osm_map` where keys match.
    """
    # find columns passed to transformers
    cat_cols = []
    num_cols = []
    for name, trans, cols in preproc.transformers_:
        if name == "cat":
            cat_cols = list(cols) if cols is not None else []
        if name == "num":
            num_cols = list(cols) if cols is not None else []

    row = {}
    # station id column
    if "StationId" in cat_cols:
        row["StationId"] = station_id

    # initialize numeric columns to zero
    for c in num_cols:
        row[c] = 0

    # populate numeric columns from osm_map: accept either item lists or comma strings
    for k, v in osm_map.items():
        # normalize value to list
        if isinstance(v, str):
            items = [s.strip() for s in v.split(",") if s.strip()]
        elif isinstance(v, (list, tuple)):
            items = list(v)
        else:
            # if numeric, set directly to value if a matching column exists
            items = None

        # possible column name patterns we set: exact key as passed, osm_count__<key>, osm_unique_count__<key>, osm_avg_count__<key>
        candidates = [k, f"osm_count__{k}", f"osm_unique_count__{k}", f"osm_avg_count__{k}"]
        for cand in candidates:
            if cand in row:
                if items is None:
                    # numeric
                    row[cand] = float(v)
                else:
                    row[cand] = len(items)
                break

    # ensure DataFrame has all columns in the same order as preproc expected when fitted
    all_cols = []
    if cat_cols:
        all_cols.extend(cat_cols)
    all_cols.extend(num_cols)

    # fill missing with zeros/empty
    for c in all_cols:
        if c not in row:
            row[c] = 0 if c in num_cols else ""

    df = pd.DataFrame([row], columns=all_cols)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to saved multi-output model joblib (e.g. model_rf_multi.joblib)")
    p.add_argument("--station-id", required=True)
    p.add_argument("--osm-json", required=True, help="JSON string or path to JSON file with OSM features mapping (key -> list or comma-string)")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        raise SystemExit(1)

    model = joblib.load(model_path)
    preproc = joblib.load(model_path.parent / "preprocessor_multi.joblib")

    osm_map = load_osm_input(args.osm_json)
    X = build_input_row(preproc, args.station_id, osm_map)

    X_p = preproc.transform(X)
    preds = model.predict(X_p)

    # print results; assume same order used in train_multi (PM2.5, PM10, AQI)
    target_names = ["PM2.5", "PM10", "AQI"]
    out = {target_names[i]: float(preds[0, i]) for i in range(min(preds.shape[1], len(target_names)))}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
