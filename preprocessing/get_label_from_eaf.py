"""
Each 10-second segment is treated as a video clip. For each clip, we determine whether a specified label occurs within that time window and export the results as a CSV file.
The `cut_videos_info.csv` file contains the names of the segmented videos, along with their corresponding start time and end time in the original video, as well as the associated EAF file.
"""

import os
import subprocess
import pandas as pd
import pympi
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


# This is path to the folder cotaining
input_folder = 'data/original_data'
csv_file = 'output/cut_videos_info.csv'
df_with_path= pd.read_csv(csv_file)

# prefix = 'test_data' # eaf column pre path
labels = ['Frontchannel','Backchannel','Utterance','Verbal','Non-Verbal','Happy','Smile',
    'Laugh','Confusion','Thinking','Surprised-Positive','Surprised-Negative','Head Tilt',
    'Head Nodding','Head shake','Agree','Disagree']

def check_backchannel(label, eaf_path, start_time_sec, end_time_sec):
    """
    Check if there is a '****' tier marked between start_time_sec and end_time_sec in the given eaf file.
    """
    start_time_ms = start_time_sec * 1000  # Convert seconds to milliseconds
    end_time_ms = end_time_sec * 1000
    
    eaf = pympi.Elan.Eaf(eaf_path)
    if label not in eaf.get_tier_names():
        return 0
    annotations = eaf.get_annotation_data_for_tier(label)
    for annotation in annotations:
        annotation_start = annotation[0]
        annotation_end = annotation[1]
        annotation_duration = annotation_end - annotation_start
        if start_time_ms <= annotation_start <= end_time_ms or start_time_ms <= annotation_end <= end_time_ms:
            if annotation_duration/1000 >= 0.1:
                return 1
    return 0

for label in labels:
    df_with_path[label] = df_with_path.apply(lambda row: check_backchannel(label, row['eaf_path'], row['start_time'], row['end_time']), axis=1)
    df_with_path.to_csv(csv_file, index=False)