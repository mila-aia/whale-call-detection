# Whale Call Detection 
[__Setup__](#setup)
| [__Data__](#data)
| [__Usage__](#usage)

This project uses Deep Learning techniques to improve whale call detection procedures based on public seismograph data. We focus on the call detections of the endangered species of Blue Whales and Fin Whales in the Lower St-Lawrence Seaway.

## Setup
### Installation
1. Download and install the latest [Anaconda Python distribution](https://www.anaconda.com/distribution/#download-section)
2. Execute the following commands to install all software requirements:
```
cd whale-call-detection
conda env create
```

3. (Optional) Install pre-commits hooks:
```
cd whale-call-detection
conda activate whale
pre-commit install
```

## Data
### Availability
The seismic data used in this study is publicly available via Natural Resoures Canada's FTP server: `ftp://ftp.seismo.nrcan.gc.ca/`. We also prepare a scrcipt for data access.

```
python scripts/download_data_ftp.py --h
usage: download_data_ftp.py [-h] [--output-dir OUTPUT_DIR] [--start-date START_DATE] [--end-date END_DATE] [--stations STATIONS] [--channels CHANNELS]

Script for downloading seismic data from CN ftp serverThe data is in MSEED format.

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        path to the output directory (default: data/)
  --start-date START_DATE
                        the starting date (yyyy-mm-dd) of the request window (default: None)
  --end-date END_DATE   the starting date (yyyy-mm-dd) of the request window (default: None)
  --stations STATIONS   the stations requested (seperated by comma) (default: PMAQ,ICQ,SNFQ,RISQ,SMQ,CNQ)
  --channels CHANNELS   the channels requested (seperated by comma) (default: HHE,HHN,HHZ,HNE,HNN,HNZ,EHZ)
```

### Pre-processing

#### Labels
The labels are generated using a matlab code. The output of this code is a matrix saved in a .mat format.
This script reads this matrix and converts the labels (date of calls, time of calls, station monitored, whale type) to the correct format.

You can choose if you want to apply a bandpass filter to the raw data by setting the flag `--bandpass_filter` to `True` or `False`.

The output are 8 possible CSVs in the `/network/projects/aia/whale_call/LABELS` folder:

`BW/`:
`bw_component_grouped_filt.csv`  `bw_filt.csv`
`bw_component_grouped_raw.csv`   `bw_raw.csv`

`FW/`:
`fw_component_grouped_filt.csv`  `fw_filt.csv`
`fw_component_grouped_raw.csv`   `fw_raw.csv`

```
usage: preprocess_labels.py [-h] [--output_dir OUTPUT_DIR] [--input_file INPUT_FILE]

Script for preprocessing the labels coming from .mat matrix

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        path to the input file (default: /network/projects/aia/whale_call/MATLAB_OUTPUT/WhaleDetectionsLSZ_new.mat)
```

## Usage

### Training

To train a 1D UNet model on a subset of dataset for testing purpose:
```
python whale/main.py fit --config experiments/mini-test.yaml
```


### Consulting MLflow logs
MLFlow logs from an `mlruns` folder located in the current directory can be consulted as follow:
```
mlflow ui
```
Otherwise, MLFlow logs can be consulted as follow:
```
mlflow ui --backend-store-uri ABSOLUTE_PATH_TO_MLRUNS_DIRECTORY
```


### Consulting Optuna logs
Optuna logs from a `optuna.sqlite3` database located in the current directory can be consulted as follow:
```
optuna-dashboard sqlite:///optuna.sqlite3
```
Otherwise, Optuna logs can be consulted as follow:
```
optuna-dashboard sqlite:///ABSOLUTE_PATH_TO_OPTUNA.SQLITE3_FILE
```