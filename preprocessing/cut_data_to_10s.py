import os
import subprocess
import pandas as pd
import time

input_folder = 'data/original_data'
output_folder = 'data/cut_data_needtobedeleted'
os.makedirs(output_folder, exist_ok=True)

start = time.time()
def get_video_duration(video_path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    return float(output)


def cut_videos(input_folder, output_folder, max_duration=300, segment_duration=10.0):
    cut_videos_info = []

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith((".mov", ".avi", ".mp4")):
                input_path = os.path.join(root, file)

                total_duration = get_video_duration(input_path)
                total_duration = min(total_duration, max_duration)

                num_segments = int(total_duration // segment_duration)

                for i in range(num_segments):
                    start_time = i * segment_duration
                    end_time = start_time + segment_duration

                    output_file = os.path.splitext(file)[0] + f"_{i+1:03d}.avi"
                    output_file_path = os.path.join(output_folder, output_file)

                    subprocess.run([
                        'ffmpeg',
                        '-ss', str(start_time),
                        '-i', input_path,
                        '-t', str(segment_duration),
                        '-strict', 'experimental',
                        output_file_path
                    ], check=True)

                    cut_video_info = {
                        'file_name': output_file,
                        'start_time': start_time,
                        'end_time': end_time
                    }
                    cut_videos_info.append(cut_video_info)

    return cut_videos_info


def add_eaf_path(row):
    # prefix = input_folder
    folder_name = '_'.join(row['file_name'].split('_')[:3])
    eaf_base = '_'.join(row['file_name'].split('_')[:4])
    eaf_name = eaf_base + '.eaf'
    return f"{input_folder}/{folder_name}/{eaf_name}"


def convert_to_dataframe(cut_videos_info):
    df = pd.DataFrame(cut_videos_info)
    df['eaf_path'] = df.apply(add_eaf_path, axis=1)
    return df


cut_videos_info = cut_videos(input_folder, output_folder, max_duration=300)

end = time.time()
print(f"Time: {end - start:.2f} seconds")
df_with_path = convert_to_dataframe(cut_videos_info)

df_with_path.to_csv(
    'output/cut_videos_info.csv',
    index=False
)
print("finish!")
# nohup python preprocessing/cut_data_to_10s.py > logs/cut_10s_time.log 2>&1 & 