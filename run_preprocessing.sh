#!/usr/bin/env bash
set -e

# =========================
# Step 0: Go to project root
# =========================
cd /home/yang/Backchannel-Detection-CCDb

echo "======================================"
echo "Step 1: Cut videos into 10-second clips"
echo "======================================"
# Input:
#   data/original_data
# Output:
#   data/cut_data
#   output/cut_videos_info.csv
python preprocessing/cut_data_to_10s.py


echo "======================================"
echo "Step 2: Generate labels from EAF"
echo "======================================"
# Input:
#   data/original_data
#   output/cut_videos_info.csv
# Output:
#   output/cut_videos_info.csv (updated with labels)
python preprocessing/get_label_from_eaf.py
# :contentReference[oaicite:0]{index=0}


echo "======================================"
echo "Step 3: Extract audio (.wav)"
echo "======================================"
# Input:
#   data/cut_data
# Output:
#   data/audio
python preprocessing/extract_wav.py
# :contentReference[oaicite:1]{index=1}


echo "======================================"
echo "Step 4: Extract OpenFace features"
echo "======================================"
# Input:
#   data/cut_data
# Output:
#   data/openface_features
python preprocessing/extract_openface_features.py
# :contentReference[oaicite:2]{index=2}


echo "======================================"
echo "Step 5: Clean features + generate label.csv"
echo "======================================"
# Input:
#   data/openface_features
#   output/cut_videos_info.csv
# Output:
#   cleaned CSV features
#   output/label.csv
python preprocessing/clean_everything.py
# :contentReference[oaicite:3]{index=3}


echo "======================================"
echo "All preprocessing finished successfully!"
echo "======================================"