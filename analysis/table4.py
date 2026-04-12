import os
import numpy as np
import pandas as pd
from pympi import Eaf

def get_files_in_folders(folder_path, file_extension='.eaf'):
    matching_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(file_extension):
                file_path = os.path.join(root, file_name)
                matching_files.append(file_path)
    return matching_files

def get_annotation_durations_for_tier(eaf_file, tier_name):
    durations = []
    eaf = Eaf(eaf_file)
    annotations = eaf.get_annotation_data_for_tier(tier_name)
    for annotation in annotations:
        start_time, end_time, _ = annotation
        duration = end_time - start_time
        durations.append(duration)
    return durations

def analyze_annotations_for_tier(folder_path, tier_name):
    total_annotations = 0
    total_duration = 0
    annotation_durations = []
    eaf_files = get_files_in_folders(folder_path)
    for eaf_file in eaf_files:
        eaf = Eaf(eaf_file)
        if tier_name in eaf.get_tier_names():
            durations = get_annotation_durations_for_tier(eaf_file, tier_name)
            total_annotations += len(durations)
            total_duration += sum(durations)
            annotation_durations.extend(durations)

    if annotation_durations:
        average_duration = round(np.mean(annotation_durations) / 1000, 2) 
        std_deviation = round(np.std(annotation_durations) / 1000, 2)  
        max_duration = round(max(annotation_durations) / 1000, 2)  
        min_duration = round(min(annotation_durations) / 1000, 2)  
    else:
        average_duration = 0
        std_deviation = 0
        max_duration = 0
        min_duration = 0

    return total_annotations, round(total_duration / 1000, 2), average_duration, std_deviation, max_duration, min_duration

def extract_and_analyze_topic_labels(folder_path, file_extension='.eaf'):
    topic_data = {}
    eaf_files = get_files_in_folders(folder_path, file_extension)

    for eaf_file in eaf_files:
        eaf = Eaf(eaf_file)

        if "Topic" in eaf.get_tier_names():
            topic_annotations = eaf.get_annotation_data_for_tier("Topic")

            for annotation in topic_annotations:
                label = annotation[2]
                start_time = annotation[0]
                end_time = annotation[1]
                duration = end_time - start_time

                if label in topic_data:
                    topic_data[label]['Count'] += 1
                    topic_data[label]['Total Duration'] += duration
                else:
                    topic_data[label] = {'Count': 1, 'Total Duration': duration}

    return topic_data

def analyze_labels_in_files(folder_path, file_extension='.eaf'):
    total_label_counts = {}
    files_with_annotations = set()

    eaf_files = get_files_in_folders(folder_path, file_extension)

    for eaf_file in eaf_files:
        eaf = Eaf(eaf_file)

        for label in eaf.get_tier_names():
            if label in total_label_counts:
                total_label_counts[label] += len(eaf.get_annotation_data_for_tier(label))
            else:
                total_label_counts[label] = len(eaf.get_annotation_data_for_tier(label))

    return total_label_counts


# Specify the target folder path
target_folder_path = 'data/original_data'

# Extract and analyze the "Topic" labels
topic_data = extract_and_analyze_topic_labels(target_folder_path)
df_topic = pd.DataFrame(topic_data).T.reset_index()
df_topic.columns = ['Label', 'Count', 'Total Duration']

# Count the total occurrences of all labels
total_label_counts = analyze_labels_in_files(target_folder_path)
df = pd.DataFrame(list(total_label_counts.items()), columns=['Label', 'Total Count'])

# Analyze annotations for each tier and output to CSV
result_data = []
tier_names = ["Frontchannel", "Backchannel", "Utterance", "Verbal", 
            "Non-Verbal", "Happy", "Smile", "Laugh", "Confusion", "Thinking", 
            "Surprised-Positive", "Surprised-Negative", "Head Tilt", "Head Nodding", 
            "Head Shake", "Agree", "Disagree", "Topic"]

for tier_name in tier_names:
    total_annotations, total_duration, average_duration, std_deviation, max_duration, min_duration = analyze_annotations_for_tier(target_folder_path, tier_name)
    result_data.append([tier_name, total_duration, total_annotations])

result_df = pd.DataFrame(result_data, columns=['Annotation', 'Duration(s)', 'Counts'])
print(result_df)