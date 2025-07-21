# Backchannel Detection on CCDb

This repository contains code for training and testing a backchannel detection model on the [CCDb dataset](https://huggingface.co/datasets/CardiffVisualComputing/CCDb).

## 📦 Setup

### Clone the repository

```bash
git clone https://github.com/hsdy1125/Backchannel-Detection-CCDb.git
cd Backchannel-Detection-CCDb
conda create -n CCDb python=3.12.2
conda activate CCDb
pip install requirements.txt
```

### Download the dataset

Download the CCDb dataset from:

[https://huggingface.co/datasets/CardiffVisualComputing/CCDb](https://huggingface.co/datasets/CardiffVisualComputing/CCDb)

Place the downloaded data into the appropriate directory as required by the feature extraction script(Get_feature.py).

## ⚒️ Feature Extraction

Run the following script to extract **visual and acoustic features** along with their corresponding labels:

```bash
python Get_feature.py
```

## 🏋️ Training

To train the model, run:

```bash
python main.py --train_mode True
```

Model checkpoints and logs will be saved to the specified `output/` directory.

## 🧪 Testing

After training, evaluate the model by running:

```bash
python main.py --train_mode False --model_path "your_model.ckpt"
```

Replace `"your_model.ckpt"` with the path to the trained model checkpoint.

## 📁 Directory Structure

```
Backchannel-Detection-CCDb/
├── Get_feature.py         # Feature extraction script
├── main.py                # Main script for training and testing
├── output/                # Output directory for logs and checkpoints
├── data/                  # Dataset directory (after download)
└── ...
```

## 📩 Contact

For any issues or questions, please open an [issue](https://github.com/hsdy1125/Backchannel-Detection-CCDb/issues) or email: dengy29@cardiff.ac.uk