#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from transformers import get_linear_schedule_with_warmup

from TransformerEncoder import CrossTransformerEncoder, TransformerEncoder
from utils import (
    generate_customized_input_npy,
    generate_acoustic_features_input_npy,
    load_and_preprocess_features,
    calcuate_metrics,
)

with open("selected_cols.txt", "r") as f:
    SELECTED_COLS = json.load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Conversation-level cross-validation training")

    parser.add_argument("--task", type=str, required=True,
                        help="Task name. Must match a real column name in output/label.csv, e.g. Backchannel, Smile, Thinking, Agree")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold_idx", type=int, required=True,
                        help="Fold index in [0, 10]")

    parser.add_argument("--feature_type", type=str, default="combined",
                        choices=["visual", "acoustic", "combined"])
    parser.add_argument("--imbalanced", type=int, default=0)

    parser.add_argument("--train_mode", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--label_csv", type=str, default="output/label.csv")
    parser.add_argument("--audio_dir", type=str, default="data/audio")
    parser.add_argument("--openface_dir", type=str, default="data/openface_features")

    parser.add_argument("--runtime_root", type=str, default="data/runtime")
    parser.add_argument("--feature_root", type=str, default="data/features")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    args = parser.parse_args()

    if not args.train_mode and not args.model_path:
        parser.error("--model_path is required when --train_mode is False")

    if args.fold_idx < 0 or args.fold_idx > 10:
        parser.error("--fold_idx must be in [0, 10]")

    return args


def extract_conversation_id(file_name: str) -> str:
    stem = os.path.splitext(str(file_name))[0]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid file_name: {file_name}")

    special_speakers = {"P19", "P20", "P21", "P22"}

    # 只要前两个 part 中任意一个属于特殊集合，就只取前两个
    if parts[0] in special_speakers or parts[1] in special_speakers:
        return "_".join(parts[:2])

    if len(parts) < 3:
        raise ValueError(f"Invalid normal conversation file_name: {file_name}")

    return "_".join(parts[:3])


def build_runtime_split(label_csv: str, seed: int, fold_idx: int):
    df = pd.read_csv(label_csv)

    if "file_name" not in df.columns:
        raise ValueError(f"{label_csv} must contain column 'file_name'")

    df = df.copy()
    df["conversation_id"] = df["file_name"].astype(str).apply(extract_conversation_id)

    conversations = sorted(df["conversation_id"].unique())

    if len(conversations) % 11 != 0:
        raise ValueError(
            f"Number of conversations = {len(conversations)}, cannot be evenly split into 11 groups."
        )

    rng = random.Random(seed)
    rng.shuffle(conversations)

    group_size = len(conversations) // 11
    groups = [conversations[i * group_size:(i + 1) * group_size] for i in range(11)]

    test_gid = fold_idx
    val_gid = (fold_idx + 1) % 11
    train_gids = [i for i in range(11) if i not in [test_gid, val_gid]]

    test_convs = set(groups[test_gid])
    val_convs = set(groups[val_gid])
    train_convs = set()
    for gid in train_gids:
        train_convs.update(groups[gid])

    train_df = df[df["conversation_id"].isin(train_convs)].copy()
    val_df = df[df["conversation_id"].isin(val_convs)].copy()
    test_df = df[df["conversation_id"].isin(test_convs)].copy()

    return train_df, val_df, test_df


def save_runtime_csvs(train_df, val_df, test_df, runtime_dir):
    os.makedirs(runtime_dir, exist_ok=True)

    drop_cols = ["conversation_id"]
    for df in [train_df, val_df, test_df]:
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    train_csv = os.path.join(runtime_dir, "train.csv")
    val_csv = os.path.join(runtime_dir, "val.csv")
    test_csv = os.path.join(runtime_dir, "test.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_csv, val_csv, test_csv


def generate_runtime_features(train_csv, val_csv, test_csv, audio_dir, openface_dir, feature_dir):
    """
    Shared features for a given (seed, fold_idx).
    Generated once and reused across tasks.
    """
    os.makedirs(feature_dir, exist_ok=True)

    split_map = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv,
    }

    for mode, csv_path in split_map.items():
        acoustic_out = os.path.join(feature_dir, f"acoustic_{mode}.npy")
        visual_out = os.path.join(feature_dir, f"visual_{mode}.npy")

        if not os.path.exists(acoustic_out):
            generate_acoustic_features_input_npy(
                labels_path=csv_path,
                input_path=audio_dir,
                input_path_ext=".wav",
                last_n_sec=10,
                output_path=acoustic_out
            )
        else:
            print(f"[Skip] acoustic feature exists: {acoustic_out}")

        if not os.path.exists(visual_out):
            generate_customized_input_npy(
                labels_path=csv_path,
                input_path=openface_dir,
                input_path_ext=".csv",
                last_n_sec=10,
                output_path=visual_out,
                selected_cols=SELECTED_COLS
            )
        else:
            print(f"[Skip] visual feature exists: {visual_out}")


def build_model(feature_type, device):
    if feature_type == "visual":
        model = TransformerEncoder(
            max_len=300,
            num_layers=1,
            input_dim=710,
            num_heads=10,
            dim_feedforward=1000,
            add_positional_encoding=True
        ).to(device)

    elif feature_type == "acoustic":
        model = TransformerEncoder(
            max_len=300,
            num_layers=1,
            input_dim=90,
            num_heads=10,
            dim_feedforward=1000,
            add_positional_encoding=True
        ).to(device)

    elif feature_type == "combined":
        model = CrossTransformerEncoder(
            max_len=300,
            num_layers=1,
            input_dim_v=710,
            input_dim_a=90,
            num_heads=10,
            dim_feedforward=1000
        ).to(device)

    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    return model


def forward_by_feature_type(model, images, feature_type, device, seq_length, input_shape):
    images = images.reshape(-1, seq_length, input_shape).to(device)

    if feature_type == "visual":
        _, outputs = model(images)

    elif feature_type == "acoustic":
        _, outputs = model(images)

    elif feature_type == "combined":
        visual = images[:, :, :710]
        acoustic = images[:, :, 710:]
        _, outputs = model(visual, acoustic)

    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    return outputs


def evaluate(model, dataloader, criterion, device, input_shape, feature_type, seq_length=300):
    model.eval()

    total_loss = 0.0
    full_predicted_list = []
    full_labels_list = []
    num_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            labels = labels.to(device).float()
            num_samples += labels.size(0)

            outputs = forward_by_feature_type(
                model=model,
                images=images,
                feature_type=feature_type,
                device=device,
                seq_length=seq_length,
                input_shape=input_shape
            )  # [B, 1]

            loss = criterion(outputs, labels.unsqueeze(1))
            total_loss += loss.item()

            predicted = (outputs > 0.5).long()
            predicted_array = predicted.squeeze().cpu().numpy()
            labels_array = labels.cpu().numpy()

            if not np.array(predicted_array).shape:
                full_predicted_list.append(predicted_array.item())
                full_labels_list.append(labels_array.item())
            else:
                full_predicted_list.extend(predicted_array.tolist())
                full_labels_list.extend(labels_array.tolist())

    avg_loss = total_loss / len(dataloader)
    acc, precision, recall, f1 = calcuate_metrics(
        np.array(full_predicted_list),
        np.array(full_labels_list),
        num_samples
    )

    return {
        "loss": round(avg_loss, 6),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    print(f"Task: {args.task}")
    print(f"Seed: {args.seed}")
    print(f"Fold: {args.fold_idx}")
    print(f"Feature type: {args.feature_type}")

    runtime_dir = os.path.join(args.runtime_root, f"seed_{args.seed}", f"fold_{args.fold_idx}")
    feature_dir = os.path.join(args.feature_root, f"seed_{args.seed}", f"fold_{args.fold_idx}")
    os.makedirs(runtime_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)


    # 1. split
    train_df, val_df, test_df = build_runtime_split(
        label_csv=args.label_csv,
        seed=args.seed,
        fold_idx=args.fold_idx
    )
    print(f"Split sizes | train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    # 2. runtime csv
    train_csv, val_csv, test_csv = save_runtime_csvs(
        train_df, val_df, test_df, runtime_dir
    )

    # 3. shared cached features
    generate_runtime_features(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        audio_dir=args.audio_dir,
        openface_dir=args.openface_dir,
        feature_dir=feature_dir
    )

    # 4. feature paths
    visual_features_path_train = os.path.join(feature_dir, "visual_train.npy")
    visual_features_path_val = os.path.join(feature_dir, "visual_val.npy")
    visual_features_path_test = os.path.join(feature_dir, "visual_test.npy")

    acoustic_features_path_train = os.path.join(feature_dir, "acoustic_train.npy")
    acoustic_features_path_val = os.path.join(feature_dir, "acoustic_val.npy")
    acoustic_features_path_test = os.path.join(feature_dir, "acoustic_test.npy")

    # 5. loaders
    input_shape, train_dataloader = load_and_preprocess_features(
        imbalanced=args.imbalanced,
        feature_type=args.feature_type,
        visual_features_path=visual_features_path_train,
        labels_csv_path=train_csv,
        acoustic_features_path=acoustic_features_path_train,
        batch_size=args.batch_size,
        task=args.task,
        shuffle=True
    )

    _, val_dataloader = load_and_preprocess_features(
        imbalanced=0,
        feature_type=args.feature_type,
        visual_features_path=visual_features_path_val,
        labels_csv_path=val_csv,
        acoustic_features_path=acoustic_features_path_val,
        batch_size=args.batch_size,
        task=args.task,
        shuffle=False
    )

    _, test_dataloader = load_and_preprocess_features(
        imbalanced=0,
        feature_type=args.feature_type,
        visual_features_path=visual_features_path_test,
        labels_csv_path=test_csv,
        acoustic_features_path=acoustic_features_path_test,
        batch_size=args.batch_size,
        task=args.task,
        shuffle=False
    )

    # 6. model
    model = build_model(args.feature_type, device)

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    seq_length = 300
    now = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

    save_dir = os.path.join(
        "output",
        "checkpoints",
        f"{args.task}_seed{args.seed}_fold{args.fold_idx}"
    )
    os.makedirs(save_dir, exist_ok=True)

    if args.train_mode:
        total_steps = args.num_epochs * len(train_dataloader)
        num_warmup_steps = int(0.1 * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )

        best_val_f1 = -1.0
        best_model_path = os.path.join(save_dir, "best_model.ckpt")
        history = []

        start_time = time.time()

        for epoch in range(args.num_epochs):
            model.train()

            training_loss = 0.0
            full_predicted_list = []
            full_labels_list = []
            num_samples = 0

            for images, labels in train_dataloader:
                labels = labels.to(device).float()
                num_samples += labels.size(0)

                outputs = forward_by_feature_type(
                    model=model,
                    images=images,
                    feature_type=args.feature_type,
                    device=device,
                    seq_length=seq_length,
                    input_shape=input_shape
                )  # [B, 1]

                loss = criterion(outputs, labels.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"[GPU] allocated={allocated:.2f}GB reserved={reserved:.2f}GB")

                training_loss += loss.item()

                predicted = (outputs > 0.5).long()
                predicted_array = predicted.squeeze().detach().cpu().numpy()
                labels_array = labels.detach().cpu().numpy()

                if not np.array(predicted_array).shape:
                    full_predicted_list.append(predicted_array.item())
                    full_labels_list.append(labels_array.item())
                else:
                    full_predicted_list.extend(predicted_array.tolist())
                    full_labels_list.extend(labels_array.tolist())

            avg_train_loss = training_loss / len(train_dataloader)
            train_acc, train_precision, train_recall, train_f1 = calcuate_metrics(
                np.array(full_predicted_list),
                np.array(full_labels_list),
                num_samples
            )

            val_result = evaluate(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                device=device,
                input_shape=input_shape,
                feature_type=args.feature_type,
                seq_length=seq_length
            )

            epoch_result = {
                "epoch": epoch,
                "train": {
                    "loss": round(avg_train_loss, 6),
                    "accuracy": train_acc,
                    "precision": train_precision,
                    "recall": train_recall,
                    "f1": train_f1
                },
                "val": val_result
            }
            history.append(epoch_result)

            print(
                f"[Epoch {epoch:03d}] "
                f"train_f1={train_f1:.3f} "
                f"val_f1={val_result['f1']:.3f} "
                f"val_acc={val_result['accuracy']:.3f}"
            )

            # save the best model rely on f1
            # if val_result["f1"] > best_val_f1:
            #     best_val_f1 = val_result["f1"]
            #     torch.save(model.state_dict(), best_model_path)
            # save the best model rely on f1
            if val_result["accuracy"] > best_val_f1:
                best_val_f1 = val_result["accuracy"]
                torch.save(model.state_dict(), best_model_path)

        elapsed_min = (time.time() - start_time) / 60.0

        model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_result = evaluate(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device,
            input_shape=input_shape,
            feature_type=args.feature_type,
            seq_length=seq_length
        )

        result = {
            "task": args.task,
            "seed": args.seed,
            "fold_idx": args.fold_idx,
            "feature_type": args.feature_type,
            "best_val_f1": best_val_f1,
            "test": test_result,
            "history": history,
            "elapsed_min": round(elapsed_min, 3),
            "timestamp": now
        }

        with open(os.path.join(save_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)

        print("\nBest validation F1:", round(best_val_f1, 3))
        print("Test result:")
        print(json.dumps(test_result, indent=2))

    else:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        test_result = evaluate(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device,
            input_shape=input_shape,
            feature_type=args.feature_type,
            seq_length=seq_length
        )

        print("Test result:")
        print(json.dumps(test_result, indent=2))


if __name__ == "__main__":
    start = time.time()
    main()

    end = time.time()
    print(f"Time: {end - start:.2f} seconds")

'''
single fold
python src/main.py --task Backchannel --seed 1234 --fold_idx 0 --feature_type combined
python src/main.py --task Backchannel --seed 1234 --fold_idx 0 --feature_type visual
python src/main.py --task Backchannel --seed 1234 --fold_idx 0 --feature_type acoustic
'''