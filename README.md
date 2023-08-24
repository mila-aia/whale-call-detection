[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/mila-aia/ner/blob/main/LICENSE)
[![ci-pipeline](https://github.com/mila-aia/whale-call-detection/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/mila-aia/whale-call-detection/actions/workflows/ci_pipeline.yml)

# Whale Call Detection 
[__Setup__](#setup)
| [__Data__](#data)
| [__Usage__](#usage)

This project uses Deep Learning techniques to improve whale call detection procedures based on public seismograph data. We focus on the call detections of the endangered species of Blue Whales and Fin Whales in the Lower St-Lawrence Seaway.

______________________________________________________________________
## Setup
### Installation
1. Download and install the latest [Anaconda Python distribution](https://www.anaconda.com/distribution/#download-section)
2. Execute the following commands to install all software requirements:
```
cd whale-call-detection
conda env create
pip install --editable .
```

3. (Optional) Install pre-commits hooks:
```
cd whale-call-detection
conda activate whale
pre-commit install
```
### Installation (Docker)
1. Download and install [docker](https://www.docker.com/).
2. Execute the following commands to install all software requirements to a Docker image:
```
cd whale-call-detection
docker build -t whale-call-detection .
```
The following commands will mount the current source code and provide access to the docker container's terminal:
```
cd whale-call-detection
docker run -it --rm \
    -v `pwd`:/home/ner \
    -p 5000:5000 \
    -p 8888:8888 \
    ner \
    /bin/bash
```
### Using the GPU from inside the Docker container
To use the GPU from within the Docker container, make sure to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and set the Docker `runtime` to `nvidia`. For example:
```
cd whale-call-detection
docker run -it --rm --runtime=nvidia \
    -v `pwd`:/home/ner \
    -p 5000:5000 \
    -p 8888:8888 \
    ner \
    /bin/bash
```

______________________________________________________________________
## Data
### Availability
The seismic data used in this study is publicly available via Natural Resoures Canada's FTP server: `ftp://ftp.seismo.nrcan.gc.ca/`. We also prepare a scrcipt for data access. Running this script is the necessary in order to download the raw data locally.

```
python scripts/download_data_ftp.py --h
usage: download_data_ftp.py [-h] [--output-dir OUTPUT_DIR] [--start-date START_DATE] [--end-date END_DATE] [--stations STATIONS] [--channels CHANNELS]

Script for downloading seismic data from CN ftp serverThe data is in MSEED format.

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        path to the output directory (default: data/RAW/)
  --start-date START_DATE
                        the starting date (yyyy-mm-dd) of the request window (default: None)
  --end-date END_DATE   the starting date (yyyy-mm-dd) of the request window (default: None)
  --stations STATIONS   the stations requested (seperated by comma) (default: PMAQ,ICQ,SNFQ,RISQ,SMQ,CNQ)
  --channels CHANNELS   the channels requested (seperated by comma) (default: HHE,HHN,HHZ,HNE,HNN,HNZ,EHZ)
```

The next step involves running the following command to list the paths to all raw sac files.

```
find data/RAW/ -name '*.SAC' -print > data/SAC_FILES_RAW.txt
```

### Pre-processing

#### Raw data 
######  Applying bandpass filter to raw 

After downloading the raw data from the ftp server, the next steps consists of applying a bandpass filter to each of these files.
We developped the script `scripts/apply_bandpass_filter.py` to do so.

```
python scripts/apply_bandpass_filter.py -h
usage: apply_bandpass_filter.py [-h] [--input-sac-files INPUT_SAC_FILES] [--output-dir OUTPUT_DIR] [--freq-min FREQ_MIN] [--freq-max FREQ_MAX]

Script to apply bandpass filter to a list of SAC files

optional arguments:
  -h, --help            show this help message and exit
  --input-sac-files INPUT_SAC_FILES
                        path to a .txt file including a list of SAC files to be filtered. (default: data/SAC_FILES_RAW.txt)
  --output-dir OUTPUT_DIR
                        path to the save filtered SAC file (default: data/FILT_10_32/)
  --freq-min FREQ_MIN   low frequency of the bandpass filter (default: 10)
  --freq-max FREQ_MAX   high frequency of the bandpass filter (default: 32)
```

The next step involves running the following command to list the paths to all filtered sac files.

```
find data/FILT_10_32/ -name '*.SAC' -print > data/SAC_FILES_FILT.txt
```

###### Remove raw data files with issues 
Some raw data `.SAC` files have some issues:
- Missing data points.
- Start date and end date different.
- Null values.

All of the files with any of these issues are removed for the training, validation and test sets.

#### Building label dataset
The labels (blue and fin whale calls) are generated using a matlab code. This code was developped by the authors of [Monitoring fin and blue whales in the Lower St. Lawrence Seaway with onshore seismometers](https://d197for5662m48.cloudfront.net/documents/publicationstatus/118893/preprint_pdf/1fb191babdbd9d518829ce1e5282a4bd.pdf). This code takes the raw data and extracts the different whale calls from it through a series of 7 predefined steps. Each of these steps are explained in this paper.

The output of this code is a matrix saved in a .mat format. This matrix is saved in this repository at `data/WhaleDetections_matlab_output.mat`. This file contains differents matrix:
- `FWC` : fin whale detections matrix.
- `BWC` : blue whale detections matrix.
- `stadir` : metadata of stations for fin whales matrix.
- `stadir_bw`: metadata of stations for blue whales matrix.

The fin whale (`FWC`) and blue whale (`BWC`) matrixes contain several information:
- `Datenum`: Date of the whale call. Matlab numerical format.
- `station_number`: ID of the station (1,2,3,4,5 or 6)
- `R`: Whale call index.
- `SNR`: Signal to Noise Ratio.
- `time_start`: Time of whale call. It is the timestamp in the whale call where the R-index is the highest.
- `num_calls_in_detection`: Number of calls in this detection.
- `detection_id`: ID of the detection.

###### Pre-processing labels

A script `scripts/preprocess_labels.py` has been built to pre-process the labels from the `.mat` file. This script reads this matrix and converts the labels (date of calls, time of calls, station monitored, whale type) to the correct format.

```
python scripts/preprocess_labels.py -h
usage: preprocess_labels.py [-h] [--input_file INPUT_FILE]

Script for preprocessing the labels coming from .mat matrix

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        path to the input file (default: data/WhaleDetections_matlab_output.mat)
```

###### Convert data types

First step includes converting the date (`DATENUM` numerical matlab format) to a `pd.DateTime` format.
Similar process is applied to the call time of detection `time_R_max`.

###### Feature engineering

Using `time_R_max` which represents the whale call time, we computed the call start and end time using the call durations for fin and blue whales we collected from the litterature ([Monitoring fin and blue whales in the Lower St. Lawrence Seaway with onshore seismometers](https://d197for5662m48.cloudfront.net/documents/publicationstatus/118893/preprint_pdf/1fb191babdbd9d518829ce1e5282a4bd.pdf)).

For fin whales, the call duration is `1` second.
For blue whales, the call duration is `8` seconds.

Using these values, we built whale calls windows for each type of whale. The whale calls windows include the call start time, the time of R max and the call end time.


###### Generate Noise Samples 
In order to build a dataset for binary classification we had to build data samples with no whale calls.

The new noise samples are built by shifting the whale call windows by 5 seconds for Fin whales and 30 seconds for blue whales.
Each of the sample pass through a quality test to check if any of them overlap with other whale calls.
Then the noise and whale call dataset are concatemated and the final dataset is shuffled.

###### Grouping components
Each seismic station records data on 3 different axis. So, for a given whale call, at each timestamp we have 3 data points. We decided to build one single value by averaging the values of these 3 data points in order to remove some noise and build a stronger signal.

###### Dataset quality 
We built datasets with different call qualities to test if the performance of our model will vary.
The call quality is determined by 2 factors:
- Whale Call Index (`R`) value.
- Signal-to-noise Ratio (`SNR`) value.

3 datasets are computed in this script:
- High Quality: `R > 5` and `SNR > 5`.
- Medium Quality: `R > 3` and `SNR > 1`.
- Low Quality : All calls.

These thresholds were decided after studying the distribution of R and SNR values accross all the different whale calls.

###### Output dataset
The final `.csv` datasets are saved under the `data/LABELS/BW/MIXED` and `data/LABELS/FW/MIXED` paths:
- `_component_grouped` extension is dataset with all 3 components grouped.
- `_filt` extension is dataset with `.sac` files where bandpass filter has been applied.
- `_raw` extension is dataset with raw `.sac` files.
- `_HQ` extension is high quality dataset .
- `_MQ` extension is high quality dataset .
- `_LQ` extension is high quality dataset .


## Usage
______________________________________________________________________
### Split data
To split the dataset into `train`, `validation` and `test` datasets we have developped a script `scripts/split_data.py`.

```
python scripts/split_data.py -h
usage: split_data.py [-h] [--input-file INPUT_FILE] [--output-path OUTPUT_PATH]

Script to apply bandpass filter to a list of SAC files

optional arguments:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE
                        Path to dataset (.csv) (default: data/LABELS/FW/MIXED/fw_HQ_component_grouped_filt.csv)
  --output-path OUTPUT_PATH
                        Path to output folder. (default: data/datasets/FWC_HQ_3CH_FILT/)
```

The dataset is split into 3 subsets using the same random seed: 80\% for the training set, 10\% for the validation set, and 10\% for the test set.

The script requires as input the PATH to the full dataset.
The output is the following `.csv` files in the output folder:
- `train.csv`
- `test.csv`
- `valid.csv`


______________________________________________________________________
### Training

- [LSTM](docs/lstm.md)
- [LSTM with hyperparameter tuning](docs/lstm-optim.md)

______________________________________________________________________
### Evaluation

______________________________________________________________________
### Consulting Optuna logs
Optuna logs from a `optuna.sqlite3` database located in the current directory can be consulted as follow:
```
optuna-dashboard sqlite:///optuna.sqlite3
```
Otherwise, Optuna logs can be consulted as follow:
```
optuna-dashboard sqlite:///ABSOLUTE_PATH_TO_OPTUNA.SQLITE3_FILE
```