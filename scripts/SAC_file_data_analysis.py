import obspy
from tqdm import tqdm
import pandas as pd

# Load labels
FILENAME = "/network/projects/aia/whale_call/SAC_FILES_RAW.txt"

with open(FILENAME) as f:
    lines = f.readlines()

files = pd.DataFrame()

for filepath in tqdm(lines):

    filepath = filepath.rstrip("\n")
    filename = filepath.split("/")[-1]
    file_date = filename

    # Load SAC file
    tr = obspy.read(filepath)

    # Get stats
    trace = tr[0]
    data = trace.data
    stats = trace.stats

    files = pd.concat(
        [
            files,
            pd.DataFrame.from_records(
                [
                    {
                        "filename": filename,
                        "filepath": filepath,
                        "starttime": stats.starttime,
                        "endtime": stats.endtime,
                        "npts": stats.npts,
                        "sampling_rate": stats.sampling_rate,
                        "delta": stats.delta,
                        "calib": stats.calib,
                        "_format": stats._format,
                        "file_date": file_date[:10],
                    }
                ]
            ),
        ]
    )


files.to_csv("/network/projects/aia/whale_call/ISSUES/file_stats.csv")
