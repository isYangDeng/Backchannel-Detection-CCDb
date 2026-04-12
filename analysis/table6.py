import pandas as pd
import pympi
import numpy as np

csv_file = 'output/cut_videos_info.csv'
df_with_path = pd.read_csv(csv_file)

def check_backchannel(eaf_path, start_time_sec, end_time_sec, occurrences):
    """
    Count whether Backchannel overlaps each 1-second bin in a 10-second clip.
    Bins are treated as:
    [0,1), [1,2), ..., [9,10)
    """
    start_time_ms = int(start_time_sec * 1000)
    end_time_ms = int(end_time_sec * 1000)

    eaf = pympi.Elan.Eaf(eaf_path)
    if 'Backchannel' not in eaf.get_tier_names():
        return

    annotations = eaf.get_annotation_data_for_tier('Backchannel')

    for annotation_start, annotation_end, _ in annotations:
        overlap_start = max(annotation_start, start_time_ms)
        overlap_end = min(annotation_end, end_time_ms)

        if overlap_start < overlap_end:
            start_second = int((overlap_start - start_time_ms) / 1000)
            end_second = int((overlap_end - start_time_ms - 1) / 1000)

            start_second = max(0, start_second)
            end_second = min(9, end_second)

            for second in range(start_second, end_second + 1):
                occurrences[second] += 1

# 10 bins: [0-1), [1-2), ..., [9-10)
occurrences = [0] * 10

for _, row in df_with_path.iterrows():
    check_backchannel(row['eaf_path'], row['start_time'], row['end_time'], occurrences)

mean_occurrences = np.mean(occurrences)
std_occurrences = np.std(occurrences)

print("Occurrences:", occurrences)
print("mean: {}  std: {}".format(mean_occurrences, std_occurrences))

for i, count in enumerate(occurrences):
    print(f"{i}-{i+1} & {count}")