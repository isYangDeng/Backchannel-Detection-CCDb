import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io.wavfile import read

import torch
from torch.utils.data import TensorDataset, DataLoader

from imblearn.over_sampling import SMOTE

from spafe.features.mfcc import mfcc
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.pncc import pncc
from spafe.features.psrcc import psrcc
from spafe.utils.preprocessing import SlidingWindow


logging.basicConfig(level=logging.INFO)


def read_sample_ids(labels_path: str):
    """
    Support both new and old formats:
    - new: file_name
    - old: id
    """
    df = pd.read_csv(labels_path)
    if "file_name" in df.columns:
        return df["file_name"].astype(str).values
    elif "id" in df.columns:
        return df["id"].astype(str).values
    else:
        raise ValueError(f"{labels_path} must contain 'file_name' or 'id'")


def pad_or_truncate_feature(feat, target_len: int, feat_dim: int):
    """
    Force temporal feature to shape [target_len, feat_dim].
    """
    if feat is None or len(feat) == 0:
        return np.zeros((target_len, feat_dim), dtype=np.float32)

    feat = np.asarray(feat, dtype=np.float32)

    if feat.ndim != 2:
        raise ValueError(f"Expected 2D feature, got shape {feat.shape}")

    if feat.shape[1] != feat_dim:
        raise ValueError(f"Feature dim mismatch: expected {feat_dim}, got {feat.shape[1]}")

    if feat.shape[0] >= target_len:
        return feat[:target_len]

    pad_len = target_len - feat.shape[0]
    last_frame = feat[-1:]
    pad = np.repeat(last_frame, pad_len, axis=0)
    return np.concatenate([feat, pad], axis=0)


def generate_customized_input_npy(
    labels_path,
    input_path,
    output_path,
    input_path_ext=".csv",
    last_n_sec=10,
    selected_cols=None
):
    """
    Generate visual features from OpenFace CSVs.
    Output shape: [N, target_len, D]
    """
    if selected_cols is None:
        selected_cols = []

    target_len = last_n_sec * 30
    sample_ids = read_sample_ids(labels_path)

    total_data = []

    for file_name in tqdm(sample_ids, desc=f"Generating visual features from {labels_path}", disable=True):
        file_path = os.path.join(input_path, file_name + input_path_ext)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OpenFace feature file not found: {file_path}")

        df_feat = pd.read_csv(file_path)
        df_feat.columns = df_feat.columns.str.strip()

        if len(selected_cols) > 0:
            missing_cols = [c for c in selected_cols if c not in df_feat.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
            df_feat = df_feat[selected_cols]

        df_feat = df_feat.fillna(0.0)
        feat = df_feat.tail(target_len).to_numpy(dtype=np.float32)
        feat_dim = feat.shape[1]
        feat = pad_or_truncate_feature(feat, target_len, feat_dim)
        total_data.append(feat)

    total_data = np.asarray(total_data, dtype=np.float32)
    np.save(output_path, total_data)
    print(f"[Visual] saved to {output_path}, shape={total_data.shape}")


def extract_acoustic_features(file_path):
    """
    Extract acoustic features with total dim = 91.
    """
    fs, sig = read(file_path)

    if sig is None or len(sig) == 0:
        logging.warning(f"Empty signal for file: {file_path}")
        return None

    # stereo -> mono
    if len(sig.shape) > 1:
        sig = np.mean(sig, axis=1)

    sig = sig.astype(np.float32)

    window = SlidingWindow(0.0666, 0.0666, "hamming")

    mfcc_feats = mfcc(sig, fs=fs, window=window)
    gfcc_feats = gfcc(sig, fs=fs, window=window)
    lfcc_feats = lfcc(sig, fs=fs, window=window)
    msrcc_feats = msrcc(sig, fs=fs, window=window)
    ngcc_feats = ngcc(sig, fs=fs, window=window)
    pncc_feats = pncc(sig, fs=fs, window=window)
    psrcc_feats = psrcc(sig, fs=fs, window=window)

    feat = np.hstack((
        mfcc_feats,
        gfcc_feats,
        lfcc_feats,
        msrcc_feats,
        ngcc_feats,
        pncc_feats,
        psrcc_feats
    )).astype(np.float32)

    return feat


def generate_acoustic_features_input_npy(
    labels_path,
    input_path,
    input_path_ext=".wav",
    last_n_sec=10,
    output_path="acoustic.npy"
):
    """
    Generate acoustic features.
    Output shape: [N, target_len, 91]
    """
    sample_ids = read_sample_ids(labels_path)
    target_len = last_n_sec * 30
    feat_dim = 91

    total_data = []

    for file_name in tqdm(sample_ids, desc=f"Generating acoustic features from {labels_path}", disable=True):
        file_path = os.path.join(input_path, file_name + input_path_ext)
        if not os.path.exists(file_path):
            logging.warning(f"Audio file not found: {file_path}, using zeros.")
            feat = np.zeros((target_len, feat_dim), dtype=np.float32)
        else:
            feat = extract_acoustic_features(file_path)
            if feat is None:
                feat = np.zeros((target_len, feat_dim), dtype=np.float32)
            else:
                feat = pad_or_truncate_feature(feat, target_len, feat_dim)

        total_data.append(feat)

    total_data = np.asarray(total_data, dtype=np.float32)
    np.save(output_path, total_data)
    print(f"[Acoustic] saved to {output_path}, shape={total_data.shape}")


def imbalanced_class(dataloader, batch_size):
    """
    Apply SMOTE on flattened features, then reshape back.
    """
    smote = SMOTE()

    x_list = []
    y_list = []

    for features, labels in dataloader:
        x_list.append(features)
        y_list.append(labels)

    X = torch.cat(x_list, dim=0).cpu().numpy()
    y = torch.cat(y_list, dim=0).cpu().numpy()

    original_shape = X.shape[1:]   # [T, D]
    X = X.reshape(X.shape[0], -1)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_resampled = torch.tensor(X_resampled, dtype=torch.float32).view(-1, *original_shape)
    y_resampled = torch.tensor(y_resampled, dtype=torch.float32)

    dataset = TensorDataset(X_resampled, y_resampled)
    dataloader_resampled = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader_resampled


def calcuate_metrics(full_predicted_list, full_labels_list, num_samples=None):
    full_predicted_list = np.asarray(full_predicted_list)
    full_labels_list = np.asarray(full_labels_list)

    TP = np.sum((full_predicted_list == 1) & (full_labels_list == 1))
    FP = np.sum((full_predicted_list == 1) & (full_labels_list == 0))
    FN = np.sum((full_predicted_list == 0) & (full_labels_list == 1))

    acc = 100.0 * np.mean(full_predicted_list == full_labels_list)
    precision = 100.0 * TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = 100.0 * TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(acc, 3), round(precision, 3), round(recall, 3), round(f1_score, 3)


def load_and_preprocess_features(
    imbalanced,
    feature_type,
    visual_features_path,
    labels_csv_path,
    acoustic_features_path,
    batch_size,
    task,
    shuffle=True
):
    """
    Load cached visual/acoustic npy features, and load labels directly from CSV.

    Return:
      - visual   -> feature shape [N, T, 710]
      - acoustic -> feature shape [N, T, 90]
      - combined -> feature shape [N, T, 800]
    """
    visual_features = np.load(visual_features_path)      # [N, T, 710]
    acoustic_features = np.load(acoustic_features_path)  # [N, T, 91]

    df = pd.read_csv(labels_csv_path)
    if task not in df.columns:
        raise ValueError(f"Task column '{task}' not found in {labels_csv_path}")
    labels = df[task].astype(np.float32).values

    visual_tensor = torch.tensor(visual_features, dtype=torch.float32)
    acoustic_tensor = torch.tensor(acoustic_features, dtype=torch.float32)

    # temporal difference
    visual_shift = torch.roll(visual_tensor, shifts=-1, dims=1)
    visual_tensor = torch.abs(visual_shift - visual_tensor)

    acoustic_shift = torch.roll(acoustic_tensor, shifts=-1, dims=1)
    acoustic_tensor = torch.abs(acoustic_shift - acoustic_tensor)

    acoustic_tensor[torch.isnan(acoustic_tensor)] = 0.0
    acoustic_tensor[torch.isinf(acoustic_tensor)] = 1.0

    # keep old behaviour: remove last acoustic dim -> 90
    acoustic_tensor = acoustic_tensor[:, :, :-1]

    if visual_tensor.shape[-1] != 710:
        raise ValueError(f"Expected visual dim 710, got {visual_tensor.shape[-1]}")
    if acoustic_tensor.shape[-1] != 90:
        raise ValueError(f"Expected acoustic dim 90, got {acoustic_tensor.shape[-1]}")

    if feature_type == "combined":
        feature_tensor = torch.cat((visual_tensor, acoustic_tensor), dim=2)   # [N, T, 800]
    elif feature_type == "visual":
        feature_tensor = visual_tensor                                         # [N, T, 710]
    elif feature_type == "acoustic":
        feature_tensor = acoustic_tensor                                       # [N, T, 90]
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(feature_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    if imbalanced == 1:
        dataloader = imbalanced_class(dataloader, batch_size)

    return feature_tensor.shape[-1], dataloader