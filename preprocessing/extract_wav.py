import os
import cv2
from moviepy import VideoFileClip
import time 
def extract_video_and_audio(input_folder, output_folder):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    error_files = []
    # Iterate over all AVI files in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith(".avi"):
            input_filepath = os.path.join(input_folder, filename)
            
            # Output filepaths
            output_audio_filepath = os.path.join(output_folder, os.path.splitext(filename)[0] + ".wav")
            try:
                # Extract audio
                video_clip = VideoFileClip(input_filepath)
                audio_clip = video_clip.audio
                audio_clip.write_audiofile(output_audio_filepath)
                
                # Close video clip
                video_clip.close()
                
                #print(f"Extracted audio to: {output_audio_filepath}")
            except Exception as e:
                error_files.append(input_filepath)
                #print(f"Error processing {input_filepath}: {e}")
                continue
    print(f"Error files: {error_files}")
# Input and output directory paths


input_folder = f"data/cut_data"
output_folder = f"data/audio"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
start_time = time.time()
extract_video_and_audio(input_folder, output_folder)
end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")

# nohup python preprocessing/extract_wav.py > logs/extract_wav.log 2>&1 &