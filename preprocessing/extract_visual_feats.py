import subprocess
import os
import time
import tempfile
import shutil

OPENFACE_BIN = os.path.expanduser("OpenFace/build/bin/FeatureExtraction")
folder_path = "data/cut_data"
output_folder = os.path.expanduser("data/openface_features")

os.makedirs(output_folder, exist_ok=True)

def convert_to_30fps(input_video, output_video):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-vf", "fps=30",
        "-r", "30",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    subprocess.run(cmd, check=True)

start_time = time.time()

with tempfile.TemporaryDirectory() as tmpdir:
    for filename in os.listdir(folder_path):
        if not filename.endswith(".avi"):
            continue

        input_path = os.path.join(folder_path, filename)
        temp_video = os.path.join(tmpdir, os.path.splitext(filename)[0] + ".mp4")

        try:
            # 1) Convert to 30 fps
            convert_to_30fps(input_path, temp_video)

            # 2) Run OpenFace, keep only CSV-related outputs
            current_command = [
                OPENFACE_BIN,
                "-f", temp_video,
                "-out_dir", output_folder,
                "-nohog",
                "-noaligned",
                "-novis",
            ]
            subprocess.run(current_command, check=True)
            print(f"Processed file: {filename}")

        except subprocess.CalledProcessError as e:
            print(f"Error processing file: {filename}, Error: {e}")

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")

# nohup python /home/yang/Backchannel-Detection-CCDb/preprocessing/extract_visual_feats.py > openface_features.log 2>&1 &
