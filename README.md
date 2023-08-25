[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/mila-aia/ner/blob/main/LICENSE)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![mypy checked](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![ci-pipeline](https://github.com/mila-aia/whale-call-detection/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/mila-aia/whale-call-detection/actions/workflows/ci_pipeline.yml)

# Whale Call Detection 
[__Overview__](#overview)
| [__Setup__](#setup)
| [__Data__](#data)
| [__Usage__](#usage)
| [__Licenses__](#licenses)

## Overview
This repository implements a framework to detect whale calls embedded in seismic waveforms. Based on seismic waveform spectrogram, the algorithm performs the following two tasks: 
1. recognizing the presence of whale calls (*classification task*) 
2. predicting call time if recognized (*regression task*).
<div align="center">
    <img src="docs/figs/overview.png" width="65%">
    <div>
    Figure 1. An overview of the framework. The data shown here is a sample of a blue whale call detected on seismic station PMAQ and the timestamp of this call is '2021-10-02 07:13:33.02'.  </div>
</div>

## Setup

### Installation
1. Download and install the latest [Anaconda Python distribution](https://www.anaconda.com/distribution/#download-section)
2. Download and uncompress the repository [here](https://github.com/mila-aia/whale-call-detection/archive/refs/heads/main.zip).
3. Execute the following commands to install all software requirements:
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

## Data
The raw seismic data used in this study is publicly available via Natural Resoures Canada's FTP server: `ftp://ftp.seismo.nrcan.gc.ca/`. The labels (blue and fin whale calls) are generated using the algorithm and code developed by [Plourde and Nedimovic [2022]](https://d197for5662m48.cloudfront.net/documents/publicationstatus/118893/preprint_pdf/1fb191babdbd9d518829ce1e5282a4bd.pdf). For more details on data availability and preprocesing, please check this [documentation](docs/data.md).

The directory format of processed waveform data is:
```
├── root_data_dir/
│   ├── 20200201/
│   │   ├── 2021.06.06.CN.CNQ..EHZ.SAC
        ├── 2021.06.06.CN.ICQ..HHE.SAC
        ├── ...
│   ├── 20200202/
│   │   ├── 2020.02.02.CN.CNQ..EHZ.SAC
        ├──2020.02.02.CN.ICQ..HHE.SAC
        ├── ...
│   ├── ...
│   │
```
The format of directory used to initialize an instance of `WhaleDataModule` is:
```
├── root_data_dir/
    ├── train.csv
    ├── valid.csv
    ├── test.csv
```

### Training
To train a Long shot-term memory (LSTM) network, please check [LSTM](docs/lstm.md) for more details.

### Making predictions (TODO)
To make prediction using a trained model: 

### WandB experiment logging (TODO)

### Consulting Optuna logs
Optuna logs from a `optuna.sqlite3` database located in the current directory can be consulted as follow:
```
optuna-dashboard sqlite:///optuna.sqlite3
```
Otherwise, Optuna logs can be consulted as follow:
```
optuna-dashboard sqlite:///ABSOLUTE_PATH_TO_OPTUNA.SQLITE3_FILE
```

## Licenses
### Models
Not applicable as no pre-trained modes are used.

### Datasets
- The seismograph data is licensed under the [Open Government License - Canada](https://open.canada.ca/en/open-government-licence-canada).

### Packages

Package | Version | License
--- | --- | ---
optuna|3.1.0|MIT License
optuna-dashboard|0.10.0| MIT License
pandas | 1.4.3  | BSD 3-Clause License
transformers | 4.20.1 | Apache 2.0 License
torchaudio | 0.12.1 | BSD 2-Clause License
torch | 1.12.1 | BSD 3-Clause License
pytorch_lightning | 1.9.2 | Apache2.0
plotly | 5.9.0 | MIT License
obspy|1.3.0 | LGPL v3.0
matplotlib | 3.6.3 | [Customized License](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE)
wget | 3.2 | GNU General Public License
types-pyyam | 6.0.12.6 | Apache 2.0 license
jsonargparse[signatures]| 4.20.0 | MIT License
wandb | 0.15.8 | MIT License

