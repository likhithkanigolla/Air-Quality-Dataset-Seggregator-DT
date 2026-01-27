# Air Quality Dataset Segregator

This project provides a Jupyter notebook to process and split air quality datasets by station, with user-defined attributes and chunked CSV outputs.

## Dataset
The dataset used in this project is taken from Kaggle.

- `station_hour.csv`: Hourly air quality measurements for various stations.
- `stations.csv`: Metadata for each station.

## Features
- Merge and filter data by StationId
- Add user-defined columns: boundary, building, geological, highway, landuse, natural
- Split filtered data into multiple CSV files (max 10,000 rows each)
- Output files are saved in a folder named after the StationId

## Usage
1. Open the `Air_Quality_Dataset_Aggregator.ipynb` notebook.
2. Set your desired input parameters (stationid, boundary, building, geological, highway, landuse, natural).
3. Run all cells.
4. Output CSV files will be created in a folder named after the station ID.

## Requirements
- Python 3.x
- pandas

## License
This project is for educational and research purposes only. Please refer to the original Kaggle dataset for licensing and usage terms.
