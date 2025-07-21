#!/usr/bin/env python
# coding: utf-8
import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime
from utils import imbalanced_class, load_and_preprocess_features,calcuate_metrics
from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from TransformerEncoder import CrossTransformerEncoder

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Encoder')

    parser.add_argument('--task', type=str, default='bc_detection',
                        help='Task name (e.g., confusion_detection, thinking_detection, agree, bc_detection)')
    parser.add_argument('--feature_type', type=str, default='combined',
                        help='Feature type (acoustic, visual, or combined)')
    parser.add_argument('--imbalanced', type=int, default=0,
                        help='Whether to apply SMOTE: 1 (yes) or 0 (no)')
    parser.add_argument('--seed', type=int, default=1111111,
                        help='Random seed for reproducibility')
    # Boolean flag for training or testing
    parser.add_argument('--train_mode', type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=True, help='Set to True for training mode, False for testing mode')
    # Path to the model checkpoint (required in testing mode)
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model checkpoint (required if train_mode is False)')
    args = parser.parse_args()
    # Enforce model_path to be provided when in testing mode
    if not args.train_mode and not args.model_path:
        parser.error('--model_path is required when --train_mode is False')

    return args

args = get_args()
task = args.task 
seed = args.seed 
feature_type = args.feature_type
imbalanced = args.imbalanced
save_tensorboard = 0
batch_size = 64
num_epochs = 60
learning_rate = 5e-5 
add_second_transf_positional_encoding=False
now = datetime.now()
current_time = now.strftime("%Y-%m-%d__%H-%M-%S")   
current_time_for_model = now.strftime("%Y-%m-%d__%H-%M")
print("Starting {}__{}_task_with_balance({}) at:".format(task,feature_type,imbalanced) +  current_time)
if save_tensorboard == 1:
    writer = SummaryWriter(f'output/tensorboard_{seed}/')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device used: {device}')

# Loading label path
labels_path_train = 'data/features/{task}_labels_train.npy'.format(task=task)
labels_path_val = 'data/features/{task}_labels_test.npy'.format(task=task)
labels_path_test = 'data/features/{task}_labels_val.npy'.format(task=task)

#Loading visual features path
visual_features_path_train = 'data/features/openface_features_train.npy'
visual_features_path_val = 'data/features/openface_features_test.npy'
visual_features_path_test = 'data/features/openface_features_val.npy'

#Loading acoustic features path
acoustic_features_path_train = 'data/features/acoustic_features_train.npy'
acoustic_features_path_val = 'data/features/acoustic_features_test.npy'
acoustic_features_path_test = 'data/features/acoustic_features_val.npy'

# Training set
input_shape, train_dataloader = load_and_preprocess_features(imbalanced,feature_type, visual_features_path_train, labels_path_train, acoustic_features_path_train, batch_size)
# print("Training set features shape:", train_dataloader.dataset.tensors[0].shape)

# Validation set
input_shape, val_dataloader = load_and_preprocess_features(0,feature_type, visual_features_path_val, labels_path_val, acoustic_features_path_val, batch_size)
# print("Validation set features shape:", val_dataloader.dataset.tensors[0].shape)
# print("Validation set labels shape:", val_dataloader.dataset.tensors[1].shape)

input_shape, test_dataloader = load_and_preprocess_features(0,feature_type, visual_features_path_test, labels_path_test, acoustic_features_path_test, batch_size)


# input_shape = 800 # change according to the features u have
seq_length = 300 #90
alpha = 0.5

model = CrossTransformerEncoder(
        max_len=300,
        num_layers=1,
        input_dim_v=710,
        input_dim_a=90,
        num_heads=10,
        dim_feedforward=1000
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0005)
criterion = nn.BCELoss()

if args.train_mode:
    # add adaptive learning rate
    total_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = 0.1 * total_steps  # 10% of the steps for warmup   #original 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    start = time.time()
    best_val_acc = 0
    best_val_f1 = 0
    for epoch in range(num_epochs):
        # print(f'****** Epoch: {epoch}/{num_epochs} ******')
        current_lr = optimizer.param_groups[0]['lr']
        # print(f"Current learning rate: {current_lr}")
        model.train()
        accuracy = 0
        num_samples = 0
        training_loss = 0
        full_predicted_list = []
        full_labels_list = []
        for i, (images, labels) in enumerate(train_dataloader):
            # print(f"current seed is {seed}; current epoch is {i}")
            images = images.reshape(-1, seq_length, input_shape).to(device)
            labels = labels.to(device)
            num_samples+=labels.size(0)

            # Forward pass
            visual, acoustic = images[:,:,:710], images[:,:,710:]
            inpp, outputs = model(visual, acoustic)
            outputs_1 =outputs[:,0,:]
            loss = criterion(outputs_1, labels.unsqueeze(1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            training_loss += loss.item()
            avg_train_loss = training_loss/len(train_dataloader)
            # print("this is training loss: {}".format(loss))
            optimizer.step()
            scheduler.step()
            # Check accuracy / mse
            predicted = (outputs_1 > 0.5).long()
            predicted_array = predicted.squeeze().cpu().numpy()
            labels_array = labels.cpu().numpy()
            if not np.array(predicted_array).shape: # If is a single number, append. If array extend
                full_predicted_list.append(predicted_array)
            else:
                full_predicted_list.extend(predicted_array)
            full_labels_list.extend(labels_array)
        full_predicted_list = np.array(full_predicted_list)
        full_labels_list = np.array(full_labels_list)
        accuracy, _, _, f1_score= calcuate_metrics(full_predicted_list, full_labels_list, num_samples)
        if save_tensorboard == 1:
            writer.add_scalars('acc/Accuracy', {'train': accuracy},epoch)
            writer.add_scalars('f1/f1_score', {'train': f1_score},epoch)
            writer.add_scalars('loss/Loss', {'train': avg_train_loss},epoch)

        # Validate the model
        # In Validation phase, we don't need to compute gradients (for memory efficiency)
        num_samples_val = 0
        model.eval()
        with torch.no_grad():
            full_predicted_val_list = []
            full_labels_val_list = []
            val_acc = 0
            val_loss = 0
            for i, (images, labels) in enumerate(val_dataloader):
                images = images.reshape(-1, seq_length, input_shape).to(device)
                labels = labels.to(device)
                num_samples_val +=labels.size(0)
                # Forward pass
                visual, acoustic = images[:,:,:710], images[:,:,710:]
                inpp, outputs = model(visual, acoustic)
                # inpp, outputs = model(images) 
                # inpp = inpp[:,1:,:]
                outputs_1 =outputs[:,0,:]            
                loss = criterion(outputs_1, labels.unsqueeze(1)) #.unsqueeze(1)
                val_loss += loss.item()
                avg_val_loss = val_loss/len(val_dataloader)
                predicted = (outputs_1 > 0.5).long()
                predicted_array = predicted.squeeze().cpu().numpy()
                # Append to lists
                labels_array = labels.cpu().numpy()
                if not np.array(predicted_array).shape: # If is a single number, append. If array extend
                    full_predicted_val_list.append(predicted_array)
                    # full_labels_val_list.append(labels_array)
                else:
                    full_predicted_val_list.extend(predicted_array)
                full_labels_val_list.extend(labels_array)
            full_predicted_val_list = np.array(full_predicted_val_list)
            full_labels_val_list = np.array(full_labels_val_list)
            val_acc, precision_val, recall_val, f1_score_val = calcuate_metrics(full_predicted_val_list, full_labels_val_list, num_samples_val)
            # print(f"current epoch is")
            print(f'Precision: {precision_val}, Recall: {recall_val}, F1 Score: {f1_score_val}, Accuracy: {val_acc}') 
            if save_tensorboard == 1:
                writer.add_scalars('acc/Accuracy', {'val': val_acc},epoch)
                writer.add_scalars('f1/f1_score', {'val': f1_score_val},epoch)
                writer.add_scalars('loss/Loss', {'val': avg_val_loss},epoch)
        
        
        if save_tensorboard == 1:
            writer.close()
        # is_best_model = (val_acc > best_val_acc)
        is_best_model = True
        if(is_best_model):
            model_name_format = 'output/checkpoint_{seed}/epoch{epoch}_{acc}_{f1}_{date}'

            # Delete previously saved model
            model_name_filled = model_name_format.format(seed = seed, epoch=epoch, acc=round(best_val_acc, 3), f1=round(best_val_f1,3),date=current_time_for_model)

            folder_path = 'output/checkpoint_{seed}'.format(seed=seed)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            try:
                os.remove(model_name_filled+'.ckpt')
            except:
                pass 
            # Do nothing if files to be deleted do not exist yet (happens in first cycle)
            # Save new best model
            best_val_acc = val_acc
            best_val_f1 = f1_score_val

            model_name_filled = model_name_format.format(seed=seed, epoch=epoch, acc=round(best_val_acc, 3), f1=round(best_val_f1,3), date=current_time_for_model)

            torch.save(model.state_dict(),model_name_filled+'.ckpt')                     

    timeTakenMin = (time.time()-start)/60
    print(f'\nFinished training in {round(timeTakenMin, 3)} min')


if args.train_mode is False:
    loaded_state = torch.load(args.model_path)
    model.load_state_dict(loaded_state)
    # Test the model
    model.eval()
    with torch.no_grad():
        full_predicted_test_list = []
        full_labels_test_list = []
        test_acc = 0
        test_loss = 0
        num_samples_test = 0
        for i, (images, labels) in enumerate(test_dataloader):
            
            images = images.reshape(-1, seq_length, input_shape).to(device)
            labels = labels.to(device)
            num_samples_test += labels.size(0)
            # Forward pass
            visual, acoustic = images[:,:,:710], images[:,:,710:]
            inpp, outputs = model(visual, acoustic)
            # inpp, outputs = model(images)
            # inpp = inpp[:,1:,:]
            outputs_1 = outputs[:,0,:]
            
            loss = criterion(outputs_1, labels.unsqueeze(1))
            # Check accuracy 
            predicted = (outputs_1 > 0.5).long()
            predicted_array = predicted.squeeze().cpu().numpy()

            # Append to lists
            labels_array = labels.cpu().numpy()
            if not np.array(predicted_array).shape:  # If is a single number, append. If array extend
                full_predicted_test_list.append(predicted_array)
            else:
                full_predicted_test_list.extend(predicted_array)
            full_labels_test_list.extend(labels_array)
            
        full_predicted_test_list = np.array(full_predicted_test_list)
        full_labels_test_list = np.array(full_labels_test_list)
        test_acc, precision_test, recall_test, f1_score_test = calcuate_metrics(full_predicted_test_list, full_labels_test_list, num_samples_test)
        print(f'Precision: {precision_test}, Recall: {recall_test}, F1 Score: {f1_score_test}, Accuracy: {test_acc}')


# python main.py