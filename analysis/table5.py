import os
import numpy as np
import pandas as pd
from pympi import Eaf

from itertools import combinations
def calculate_pairwise_duration_ratio(eaf_files, base_tier, target_tiers):
    total_duration_per_tier = {tier: 0 for tier in target_tiers}
    pairwise_overlap = {(t1, t2): 0 for t1, t2 in combinations(target_tiers, 2)}

    for eaf_file in eaf_files:
        base_annotations = get_annotation_data(eaf_file, base_tier)
        tier_annotations = {tier: get_annotation_data(eaf_file, tier) for tier in target_tiers}

        for base_start, base_end, _ in base_annotations:
            for tier in target_tiers:
                for s, e, _ in tier_annotations[tier]:
                    overlap_start = max(base_start, s)
                    overlap_end = min(base_end, e)
                    if overlap_start < overlap_end:
                        total_duration_per_tier[tier] += overlap_end - overlap_start

            for t1, t2 in combinations(target_tiers, 2):
                for s1, e1, _ in tier_annotations[t1]:
                    for s2, e2, _ in tier_annotations[t2]:
                        overlap_start = max(base_start, s1, s2)
                        overlap_end = min(base_end, e1, e2)
                        if overlap_start < overlap_end:
                            pairwise_overlap[(t1, t2)] += overlap_end - overlap_start

    pairwise_ratio = []
    for (t1, t2), overlap_dur in pairwise_overlap.items():
        base_dur = total_duration_per_tier[t1]
        ratio = (overlap_dur / base_dur) if base_dur > 0 else 0
        pairwise_ratio.append({
            "Tier 1": t1,
            "Tier 2": t2,
            "Overlap Duration (s)": round(overlap_dur / 1000, 2),
            "Tier 1 Total Duration (s)": round(base_dur / 1000, 2),
            "Overlap Ratio": round(ratio, 4)
        })
    return pd.DataFrame(pairwise_ratio)

def get_files_in_folders(folder_path, file_extension='.eaf'):
    matching_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(file_extension):
                file_path = os.path.join(root, file_name)
                matching_files.append(file_path)
    return matching_files

def get_annotation_data(eaf_file, tier_name):
    eaf = Eaf(eaf_file)
    if tier_name in eaf.get_tier_names():
        return eaf.get_annotation_data_for_tier(tier_name)
    return []

def calculate_cooccurrence(eaf_files, base_tier, target_tiers):
    cooccurrence_data = {tier: {'Count': 0, 'Total Duration': 0} for tier in target_tiers}

    for eaf_file in eaf_files:
        base_annotations = get_annotation_data(eaf_file, base_tier)
        for target_tier in target_tiers:
            target_annotations = get_annotation_data(eaf_file, target_tier)
            for base_annotation in base_annotations:
                base_start, base_end, _ = base_annotation
                for target_annotation in target_annotations:
                    target_start, target_end, _ = target_annotation
                    overlap_start = max(base_start, target_start)
                    overlap_end = min(base_end, target_end)
                    if overlap_start < overlap_end:  # Check if there is an overlap
                        overlap_duration = overlap_end - overlap_start
                        cooccurrence_data[target_tier]['Count'] += 1
                        cooccurrence_data[target_tier]['Total Duration'] += overlap_duration

    # Convert duration from milliseconds to seconds and round to two decimal places
    for tier in cooccurrence_data:
        cooccurrence_data[tier]['Total Duration'] = round(cooccurrence_data[tier]['Total Duration'] / 1000, 2)
        
    return cooccurrence_data

# Specify the target folder path and the tiers to analyze
target_folder_path = 'data/original_data'
base_tier = 'Backchannel'
target_tiers = ["Head Nodding", "Head Shake", "Head Tilt", 
                "Smile", "Laugh", "Surprised-Positive", "Surprised-Negative", "Confusion", "Thinking",
                "Verbal", "Non-Verbal" ]

eaf_files = get_files_in_folders(target_folder_path)

# Calculate cooccurrence data
cooccurrence_data = calculate_cooccurrence(eaf_files, base_tier, target_tiers)

# Convert to DataFrame for easier plotting
df_cooccurrence = pd.DataFrame(cooccurrence_data).T.reset_index()
df_cooccurrence.columns = ['Annotation', 'Count', 'Duration (s)']

print(df_cooccurrence)
