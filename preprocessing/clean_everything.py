import os
import shutil

# Replace with the folder path
folder_path = "data/openface_features"
# Traverse all files and subfolders
for root, dirs, files in os.walk(folder_path, topdown=False):
    # Delete all non-CSV files
    for file in files:
        if not file.endswith('.csv'):
            file_path = os.path.join(root, file)
            print(f"Deleting file: {file_path}")
            os.remove(file_path)
    # Delete all subfolders and their contents
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        print(f"Deleting folder: {dir_path}")
        shutil.rmtree(dir_path)
print("Cleanup completed!")


import pandas as pd
# Read original CSV
input_path = "output/cut_videos_info.csv"
output_path = "output/label.csv"
df = pd.read_csv(input_path)

# 1️⃣ Remove ".avi" from the first column
df.iloc[:, 0] = df.iloc[:, 0].str.replace('.avi', '', regex=False)
# 2️⃣ Remove columns 2, 3, 4 (index starts from 0)
df = df.drop(df.columns[[1, 2, 3]], axis=1)

# 3️⃣ Clean column names: replace '-' and ' ' with '_'
df.columns = [col.replace('-', '_').replace(' ', '_') for col in df.columns]

# 4️⃣ Save cleaned file
df.to_csv(output_path, index=False)
print(f"Saved cleaned file to: {output_path}")