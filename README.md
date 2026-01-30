**AP001 Model Training (quick starter)**

Files added in `src/`:
- `data_loader.py` : reads station chunk CSVs and extracts numeric OSM-derived features from `station_osm_features_filtered_110.csv` and optionally `station_locations.csv`.
- `train.py` : trains three models (LinearRegression, RandomForest, GradientBoosting), evaluates and saves models and metrics.
- `predict.py` : loads a saved model and predicts a representative air-quality value for a station.

# Air Quality Dataset Aggregator — Usage

This repository trains multi-output regressors to predict PM2.5, PM10, and AQI from station data and OSM-derived features.

Quick layout
- `output/` — station chunk CSVs (e.g., `output/AP001/*.csv`).
- `src/` — training and prediction scripts.
	- `src/train_multi.py` — train multi-output models (uses row-level Datetime features if present).
	- `src/predict_input.py` — predict using saved models; supports loading all `model_*_multi.joblib` files from a models directory.
	- `src/data_loader_chunks.py` — helpers to read/aggregate chunk CSVs.
- `models_multi/` — model artifacts saved by `train_multi.py`.

Requirements
- Python 3.8+
- Install dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1    # Windows PowerShell
.venv\\Scripts\\activate.bat    # Windows cmd
pip install -r requirements.txt
```

Training (multi-output)

```bash
python -m src.train_multi --station-folder output/AP001 --out-dir models_multi/AP001_datetime
```

Predicting from JSON input
- Create a JSON with OSM lists and `datetime` (optional). Example `input_osm.json`:

```json
{
	"datetime": "2017-11-25 18:00:00",
	"amenity": ["parking","school"],
	"highway": ["bus_stop"]
}
```

- Run prediction (prints predictions for all models in the models folder):

```bash
python -m src.predict_input --models-dir models_multi/AP001_datetime --preproc models_multi/AP001_datetime/preprocessor_multi.joblib --station-id AP001 --osm-json input_osm.json
```

Notes and recommendations
- `Datetime` is now used to derive `dt_hour`, `dt_dayofweek`, `dt_month`, and `dt_is_weekend` features for row-level training.
- If you want better temporal modeling, consider adding cyclical encoding for hour (sin/cos) and retraining.
- To cleanly remove files, verify changes before deleting; placeholders were used for removed debug/test files.

Contact
- Ask me to run training/predictions or to further clean up files (I can permanently delete files if you confirm).
---

Getting started

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train multi-output models (uses `Datetime` when present to derive time features):

```powershell
python -m src.train_multi --station-folder output/AP001 --out-dir models_multi/AP001_datetime
```

3. Predict from a JSON input (prints predictions from all saved models in the models directory):

```powershell
python -m src.predict_input --models-dir models_multi/AP001_datetime --preproc models_multi/AP001_datetime/preprocessor_multi.joblib --station-id AP001 --osm-json input_osm.json
```

`input_osm.json` example:

```json
{
	"datetime": "2017-11-25 18:00:00",
	"amenity": ["parking","school"],
	"highway": ["bus_stop"]
}
```

Notes
- The pipeline supports two modes: row-level training (preferred when `Datetime` exists in chunk CSVs) and station-level aggregation (fallback).
- Datetime-derived features added: `dt_hour`, `dt_dayofweek`, `dt_month`, `dt_is_weekend`.
- If you want stronger temporal modeling, consider adding cyclical encoding for `dt_hour` or additional meteorological features.

Files of interest
- `src/train_multi.py` — trains multi-output models and saves preprocessor + models to the specified `--out-dir`.
- `src/predict_input.py` — loads `preprocessor_multi.joblib` and all `model_*_multi.joblib` files in a models directory and prints predictions for each model.
- `src/data_loader_chunks.py` — utilities to read chunk CSVs and extract OSM-derived features.

If you'd like, I can also:
- Save predictions to CSV,
- Retrain with hyperparameter tuning,
- Add cyclical encoding for hours and retrain.

