import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import csv
import time
import logging
from tqdm import tqdm
import pandas as pd 
from spafe.features.mfcc import mfcc
from spafe.features.bfcc import bfcc
from spafe.features.cqcc import cqcc
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.pncc import pncc
from spafe.features.psrcc import psrcc
from scipy.io.wavfile import read
from spafe.utils.preprocessing import SlidingWindow
def generate_customized_input_npy(labels_path,#="Backchannel_detection/input_data/original/bc_detection_train.csv",
                                  input_path,#='../BackchannelAgreementEstimation/MMEmotionRecognition/data/inputData-SSP-Backchannel/openface/AUs/',
                                  output_path,#='Backchannel_detection/input_data/npy/agreement_openface_features_674_90.npy',
                                  input_path_ext = '.csv',
                                  last_n_sec = 10, #10
                                  selected_cols = []):
    pick_last_n_cols = last_n_sec * 30
    # read labels file names 
    df_labels = pd.read_csv(filepath_or_buffer=labels_path, sep =",")['id'].values
    # Loop
    total_data = []
    for i, file_name in enumerate(tqdm(df_labels)):
        file_path = os.path.join(input_path, file_name + input_path_ext)
        # Read CSV
        df_aus = pd.read_csv(filepath_or_buffer=file_path, sep =",")
        
        df_aus.columns = df_aus.columns.str.strip() # Remove spaces from column names
        # Selected Columns
        if len(selected_cols) != 0:
            df_aus = df_aus[selected_cols]
        df_aus = df_aus.tail(pick_last_n_cols) 
        #print(df_aus.shape)
        df_aus = df_aus.fillna(0) # Fill nans with zeros if exists
        # Append
        total_data.append(df_aus.to_numpy())
    total_data = np.array(total_data)
    # print("labels shape", df_labels.shape) #labels shape (1974,)  labels shape (247,)
    # print("input shape", total_data.shape, "\n")#input shape (1974, 300, 671)  input shape (247, 300, 671) 
    np.save(output_path,total_data)


def generate_labels_input_npy(labels_path,#='Backchannel_detection/input_data/original/bc_detection_{mode}.csv',
                                  output_path,#='Backchannel_detection/input_data/npy/detection_custom_labels_{mode}_{length}.npy',
                                  modes,#=['train', 'val', 'test'],
                                  task,# = 'thinking')
):
    for mode in modes:
        # read labels file names 
        df_labels = pd.read_csv(filepath_or_buffer=labels_path.format(mode=mode,task=task), sep =",")['label'].values
        # print("labels shape", df_labels.shape, "\n")
        np.save(output_path.format(mode=mode, task = task, length=len(df_labels)),df_labels)
        

# Define a function to extract acoustic features from audio files using Spafe
def extract_acoustic_features(file_path):
    fs, sig = read(file_path)
    duration = len(sig) / fs
    print(f"Duration: {duration}")
    if len(sig) == 0:
        logging.warning(f"Empty signal for file: {file_path}")
        return None
    
    # Extract acoustic features 
    # how to calcuate the win_len and win_hop 凑 acoustic features shape to 300,91 
    #a/f - w = 300*s  s:win_hop; w:win_len; a:signal_length=len(sig); f:hz
    # normally win_len = 0.025s, win_hop = 0.01s get 2000 features
    mfcc_feats = mfcc(sig, fs=fs, window=SlidingWindow(0.0666, 0.0666, "hamming")) 
    gfcc_feats = gfcc(sig, fs=fs, window=SlidingWindow(0.0666, 0.0666, "hamming"))
    lfcc_feats = lfcc(sig, fs=fs, window=SlidingWindow(0.0666, 0.0666, "hamming"))
    msrcc_feats = msrcc(sig, fs=fs, window=SlidingWindow(0.0666, 0.0666, "hamming"))
    ngcc_feats = ngcc(sig, fs=fs, window=SlidingWindow(0.0666, 0.0666, "hamming"))
    pncc_feats = pncc(sig, fs=fs, window=SlidingWindow(0.0666, 0.0666, "hamming"))
    psrcc_feats = psrcc(sig, fs=fs, window=SlidingWindow(0.0666, 0.0666, "hamming"))
    return np.hstack((mfcc_feats, gfcc_feats, lfcc_feats,
                        msrcc_feats, ngcc_feats,
                        pncc_feats, psrcc_feats))

# Modify the generate_acoustic_features_input_npy function to utilize the extract_acoustic_features function
def generate_acoustic_features_input_npy(labels_path,
                                         input_path,
                                         input_path_ext,
                                         last_n_sec,
                                         output_path):
    
    df_labels = pd.read_csv(labels_path)['id'].values
    total_data = []
    max_length = 300  # Maximum length of the audio files  should be if 10s 300 
    # non_standard_files = []  # Store files with non-standard shape
    for i, file_name in enumerate(tqdm(df_labels)):
        file_path = os.path.join(input_path, file_name + input_path_ext)

        # Extract acoustic features from the audio file
        audio_features = extract_acoustic_features(file_path)
        if audio_features is None:
            audio_features = np.zeros((max_length, 91))
        if len(audio_features) < max_length:
            # If audio features length is less than max_length, replicate the last frame to fill to max_length
            last_frame = audio_features[-1]
            audio_features = np.vstack([audio_features] * (max_length // len(audio_features)))
            remainder = max_length % len(audio_features)
            audio_features = np.vstack([audio_features, np.tile(last_frame, (remainder, 1))])
            # print(f'Modified audio_features shape: {audio_features.shape}')
        elif len(audio_features) > max_length:
            audio_features = audio_features[:max_length, :]
        # print(f'Modified audio_features shape: {audio_features.shape}')
        total_data.append(audio_features)

    # print("Files with non-standard shape(2004,91):", non_standard_files)

    total_data1 = np.array(total_data)
    
    print("Labels shape:", df_labels.shape)
    print("Input shape:", total_data1.shape)
    
    # Save the numpy array
    np.save(output_path, total_data1)


def imbalanced_class(dataloader):
    smote = SMOTE()
    # oversample with smote for imbalanced data
    # 获取训练集数据
    X_train = [images for images, _ in dataloader]
    y_train = [labels for _, labels in dataloader]
    # 将列表转换为numpy数组
    X_train_resampled = torch.cat(X_train, dim=0).cpu().numpy()
    y_train_resampled = torch.cat(y_train, dim=0).cpu().numpy()
    # Reshape to 2D arrays
    original_shape = X_train_resampled[0].shape  # Assuming all samples have the same shape
    X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], -1)  # Flattens each image
    
    # Apply SMOTE
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_resampled, y_train_resampled)
    # sum(sum(1 for element in tensor.flatten() if element.item() == 0) for tensor in y_train_resampled)
    # check the shape of the resampled data on watch board
    
    # Recover original shape
    X_train_resampled = torch.Tensor(X_train_resampled).view(-1, *original_shape)

    resampled_dataset = TensorDataset(torch.Tensor(X_train_resampled), torch.Tensor(y_train_resampled))
    resampled_dataloader = DataLoader(resampled_dataset, batch_size=batch_size, shuffle=True)
    return resampled_dataloader

def calcuate_metrics(full_predicted_list, full_labels_list, num_samples):
    # Assuming you have true positives (TP), false positives (FP), and false negatives (FN) counts
    TP = np.sum((full_predicted_list == full_labels_list) & (full_predicted_list == 1))
    FP = np.sum((full_predicted_list != full_labels_list) & (full_predicted_list == 1))
    FN = np.sum((full_predicted_list != full_labels_list) & (full_predicted_list == 0))
    # acc = (accuracy_0 + acuuracy_1)/2
    
    acc = 100 * ((full_predicted_list== full_labels_list).sum()) /num_samples
    # Precision
    precision = (TP / (TP + FP))*100
    recall = (TP / (TP + FN))*100
    f1_score = 2 * (precision * recall) / (precision + recall)
    return round(acc,3), round(precision,3), round(recall,3), round(f1_score,3)


def load_and_preprocess_features(imbalanced, feature_type, features_path, labels_path, features_voice_path, batch_size):
    # Load features and labels
    visual_features = np.load(features_path)
    labels = np.load(labels_path)
    features_voice = np.load(features_voice_path)
    
    # Preprocess features
    visual_features_tensor = torch.Tensor(visual_features)
    visual_features_tensor_s = torch.roll(visual_features_tensor, shifts=-1, dims=1)
    visual_features_tensor = abs(visual_features_tensor_s - visual_features_tensor)
    
    
    acoustic_features_tensor = torch.Tensor(features_voice)
    acoustic_features_tensor_s = torch.roll(acoustic_features_tensor, shifts=-1, dims=1)
    acoustic_features_tensor = abs(acoustic_features_tensor_s - acoustic_features_tensor)

    acoustic_features_tensor[torch.isnan(acoustic_features_tensor)] = 0.0
    acoustic_features_tensor[torch.isinf(acoustic_features_tensor)] = 1
    acoustic_features_tensor = acoustic_features_tensor[:, :, :-1]
        
    # Combine features
    features_tensor_mix = torch.cat((visual_features_tensor, acoustic_features_tensor), dim=2)
    labels_tensor = torch.Tensor(labels.astype(np.float64))
    
    # Visual features
    dataset_visual = TensorDataset(visual_features_tensor, labels_tensor)
    dataloader_visual = DataLoader(dataset_visual, batch_size=batch_size, shuffle=True)
    
    # Acoustic features
    dataset_acoustic = TensorDataset(acoustic_features_tensor, labels_tensor)
    dataloader_acoustic = DataLoader(dataset_acoustic, batch_size=batch_size, shuffle=True)
    
    # Combined features
    dataset = TensorDataset(features_tensor_mix, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if feature_type == 'visual':
        if imbalanced == 1:
            dataloader_visual_resample = imbalanced_class(dataloader_visual)
            return 710, dataloader_visual
        return 710, dataloader_visual
    if feature_type == 'acoustic':
        if imbalanced == 1:
            dataloader_acoustic_resample = imbalanced_class(dataloader_acoustic)
            return 90, dataloader_acoustic
        return 90, dataloader_acoustic
    if feature_type == 'combined':
        if imbalanced == 1:
            dataloader_resample = imbalanced_class(dataloader)
            return 800, dataloader
        return 800, dataloader
