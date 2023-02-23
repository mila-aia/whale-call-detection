import obspy
from tqdm import tqdm
import pandas as pd

# Load labels
raw_calls = pd.read_csv(
    "/network/projects/aia/whale_call/LABELS/FW/fw_raw.csv"
)

# Create empty df for results
identified_samples = pd.DataFrame()

for _, row in tqdm(raw_calls.iterrows()):

    # Convert start and end times
    start_time = obspy.UTCDateTime(row["time_window_start"])
    end_time = obspy.UTCDateTime(row["time_window_end"])

    # Load SAC file
    tr = obspy.read(row["file_path"], starttime=start_time, endtime=end_time)

    # Get stats
    trace = tr[0]
    data = trace.data
    stats = trace.stats
    min = data.min()
    max = data.max()
    mean = data.mean()

    if (max - min == 0) | (mean == 0):

        identified_samples = identified_samples.append(
            {
                "filename": row["file_path"],
                "starttime": stats.starttime,
                "endtime": stats.endtime,
                "npts": stats.npts,
                "sampling_rate": stats.sampling_rate,
                "min": min,
                "max": max,
                "mean": mean,
            },
            ignore_index=True,
        )


identified_samples.to_csv(
    "/network/projects/aia/whale_call/ISSUES/fw_0_values.csv"
)
