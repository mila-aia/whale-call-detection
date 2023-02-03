# Whale Call Detection 

This project uses Deep Learning techniques to improve whale call detection procedures based on public seismograph data. We focus on the call detections of the endangered species of Blue Whales and Fin Whales in the Lower St-Lawrence Seaway.

## Installation
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
The seismic data used in this study is publicly available via Natural Resoures Canada's FTP server: `ftp://ftp.seismo.nrcan.gc.ca/wfdata6/`. We also prepare a scrcipt for data access.

```
python scripts/download_data_ftp.py --h
usage: download_data_ftp.py [-h] [--output-dir OUTPUT_DIR] [--start_date START_DATE] [--end_date END_DATE] [--stations STATIONS]
                            [--channels CHANNELS]

Script for downloading seismic data from CN ftp serverThe data is in MSEED format.

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        path to the output directory (default: data/)
  --start_date START_DATE
                        the starting date (yyyy-mm-dd) of the request window (default: None)
  --end_date END_DATE   the starting date (yyyy-mm-dd) of the request window (default: None)
  --stations STATIONS   the stations requested (seperated by comma) (default: None)
  --channels CHANNELS   the channels requested (seperated by comma) (default: HHE,HHN,HHZ,HNE,HNN,HNZ,EHZ)
```