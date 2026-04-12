# CCDb: Backchannel Detection Benchmark

This repository is the official implementation of the paper:

**“CCDb+: Enhanced Annotations and Multi-Modal Benchmark for Natural Dyadic Conversations”**  
Accepted at ACM Multimedia 2025.  
Paper link: https://dl.acm.org/doi/10.1145/3746027.3755333

---

## Overview

This repository provides:

- A full preprocessing pipeline for CCDb+ data
- Multi-modal feature extraction (audio + visual)
- Training and evaluation scripts for:
  - Transformer-based models
  - SVM baselines
- Support for both **single-fold** and **11-fold cross-validation**

---

## Reproducibility

There are **two ways** to reproduce the results.

---

### Option 1: From Original Data (Full Pipeline)

Download the original dataset:

👉 https://huggingface.co/datasets/CardiffVisualComputing/CCDb

Then run:

```bash
bash run_preprocessing.sh
bash run_model.sh
````

This will:

1. Segment videos into 10-second clips
2. Generate labels from EAF annotations
3. Extract audio features
4. Extract OpenFace visual features
5. Clean data and generate `label.csv`
6. Train and evaluate models

---

### Option 2: From Preprocessed Data (Recommended)

Download preprocessed data:

👉 [https://huggingface.co/datasets/isYang66/CCDb](https://huggingface.co/datasets/isYang66/CCDb)

Then directly run:

```bash
bash run_model.sh
```

This skips all preprocessing steps.

---

## Running Models

Examples:

```bash
# Transformer (single fold)
bash run_model.sh --model transformer --mode single --task Backchannel --fold_idx 0 --seed 1 --feature_type combined

# Transformer (11-fold CV)
bash run_model.sh --model transformer --mode cv --task Backchannel --seed 1 --feature_type combined

# SVM (single fold)
bash run_model.sh --model svm --mode single --task Backchannel --fold_idx 0 --seed 1 --feature_type visual

# SVM (11-fold CV)
bash run_model.sh --model svm --mode cv --task Backchannel --seed 1 --feature_type acoustic
```

---

## Environment Setup

Create environment:

```bash
conda create -n ccdb python=3.9 -y
conda activate ccdb
```

Install dependencies:

```bash
pip install pandas pympi-ling moviepy
conda install -c conda-forge opencv
pip install imbalanced-learn
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy pandas scipy scikit-learn tqdm spafe matplotlib
python -m pip install transformers
pip install spafe
```

---

## Additional Requirements (Preprocessing Only)

If you want to run `run_preprocessing.sh`, you must install **OpenFace**.

Official guide:
[https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation)

### System Dependencies

```bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install g++-8
sudo apt-get install cmake
sudo apt-get install libopenblas-dev

sudo apt-get install git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python3-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev
```

---

### Install OpenCV (required by OpenFace)

```bash
wget https://github.com/opencv/opencv/archive/4.1.0.zip
sudo unzip 4.1.0.zip
cd opencv-4.1.0
sudo mkdir build
cd build

sudo cmake -D CMAKE_BUILD_TYPE=RELEASE \
           -D CMAKE_INSTALL_PREFIX=/usr/local \
           -D BUILD_TIFF=ON \
           -D WITH_TBB=OFF ..

sudo make -j2
sudo make install
sudo ldconfig
cd ../..
```

---

### Install dlib

```bash
wget http://dlib.net/files/dlib-19.13.tar.bz2
tar xf dlib-19.13.tar.bz2
cd dlib-19.13
sudo mkdir build
cd build
sudo cmake ..
sudo cmake --build . --config Release
sudo make install
sudo ldconfig
cd ../..
```

---

### Install OpenFace

```bash
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
mkdir build
cd build

sudo cmake -D CMAKE_CXX_COMPILER=/usr/bin/g++ \
           -D CMAKE_C_COMPILER=/usr/bin/gcc \
           -D CMAKE_BUILD_TYPE=RELEASE ..

make -j2
```

---

### Download Required Model Files

Download from:

👉 [https://www.dropbox.com/sh/o8g1530jle17spa/AADRntSHl_jLInmrmSwsX-Qsa?dl=0](https://www.dropbox.com/sh/o8g1530jle17spa/AADRntSHl_jLInmrmSwsX-Qsa?dl=0)

Place the four `.dat` files into:

```bash
~/opencv-4.1.0/build/OpenFace/build/bin/model/patch_experts/
```

---

### Test OpenFace

```bash
~/opencv-4.1.0/build/OpenFace/build/bin/FeatureExtraction \
  -f "/home/yang/opencv-4.1.0/samples/0_train_rec18_pos3_video.avi" \
  -out_dir "$HOME/face"
```

---

## Notes

* Preprocessing is **time-consuming** (OpenFace extraction is the bottleneck)
* Using preprocessed data is strongly recommended for quick reproduction
* All experiments support:

  * multi-modal features (visual / acoustic / combined)
  * 11-fold cross-validation
  * multiple seeds

---

## Citation

If you use this repository, please cite the original paper:

```
@inproceedings{deng2025ccdb+,
  title={C{CD}b+: {Enhanced Annotations and Multi-Modal Benchmark for Natural Dyadic Conversations}},
  author={Deng, Yang and Lai, Yu-Kun and Rosin, Paul L},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={5657--5666},
  year={2025}
}
```
