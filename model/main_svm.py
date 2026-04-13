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

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import (
    generate_customized_input_npy,
    generate_acoustic_features_input_npy,
)

with open("selected_cols.txt", "r") as f:
    SELECTED_COLS = json.load(f)
   

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def inspect_array(name, X):
    print(f"\nInspecting {name}")
    print("  shape:", X.shape)
    print("  dtype:", X.dtype)

    finite_mask = np.isfinite(X)
    num_nonfinite = (~finite_mask).sum()
    print("  non-finite count:", int(num_nonfinite))

    if num_nonfinite > 0:
        print("  has_nan:", np.isnan(X).any())
        print("  has_posinf:", np.isposinf(X).any())
        print("  has_neginf:", np.isneginf(X).any())

    finite_vals = X[finite_mask]
    if finite_vals.size > 0:
        print("  finite min:", finite_vals.min())
        print("  finite max:", finite_vals.max())

def parse_args():
    parser = argparse.ArgumentParser(description="Conversation-level CV with Linear SVM")

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold_idx", type=int, required=True)

    parser.add_argument("--feature_type", type=str, default="combined",
                        choices=["visual", "acoustic", "combined"])
    parser.add_argument("--imbalanced", type=int, default=0)

    parser.add_argument("--label_csv", type=str, default="output/label.csv")
    parser.add_argument("--audio_dir", type=str, default="data/audio")
    parser.add_argument("--openface_dir", type=str, default="data/openface_features")

    parser.add_argument("--runtime_root", type=str, default="data/runtime")
    parser.add_argument("--feature_root", type=str, default="data/features")

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=5000)

    args = parser.parse_args()

    if args.fold_idx < 0 or args.fold_idx > 10:
        parser.error("--fold_idx must be in [0, 10]")

    return args


def extract_conversation_id(file_name: str) -> str:
    stem = os.path.splitext(str(file_name))[0]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid file_name: {file_name}")

    special_speakers = {"P19", "P20", "P21", "P22"}

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

    for df in [train_df, val_df, test_df]:
        if "conversation_id" in df.columns:
            df.drop(columns=["conversation_id"], inplace=True)

    train_csv = os.path.join(runtime_dir, "train.csv")
    val_csv = os.path.join(runtime_dir, "val.csv")
    test_csv = os.path.join(runtime_dir, "test.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_csv, val_csv, test_csv


def generate_runtime_features(train_csv, val_csv, test_csv, audio_dir, openface_dir, feature_dir):
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


def load_labels(csv_path: str, task: str):
    df = pd.read_csv(csv_path)
    if task not in df.columns:
        raise ValueError(f"Task '{task}' not found in {csv_path}")
    y = df[task].astype(np.int64).to_numpy()
    return y


def load_features(feature_type, visual_path, acoustic_path):
    visual = np.load(visual_path) if os.path.exists(visual_path) else None
    acoustic = np.load(acoustic_path) if os.path.exists(acoustic_path) else None

    if feature_type == "visual":
        X = visual
    elif feature_type == "acoustic":
        X = acoustic
    else:
        if visual is None or acoustic is None:
            raise ValueError("Combined features require both visual and acoustic npy files.")
        X = np.concatenate([visual, acoustic], axis=-1)

    if X.ndim != 3:
        raise ValueError(f"Expected 3D feature array, got shape {X.shape}")

    X = X.reshape(X.shape[0], -1).astype(np.float32)
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    return X


def evaluate(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Task: {args.task}")
    print(f"Seed: {args.seed}")
    print(f"Fold: {args.fold_idx}")
    print(f"Feature type: {args.feature_type}")

    runtime_dir = os.path.join(args.runtime_root, f"svm_seed_{args.seed}", f"fold_{args.fold_idx}")
    feature_dir = os.path.join(args.feature_root, f"svm_seed_{args.seed}", f"fold_{args.fold_idx}")
    os.makedirs(runtime_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    train_df, val_df, test_df = build_runtime_split(
        label_csv=args.label_csv,
        seed=args.seed,
        fold_idx=args.fold_idx
    )
    print(f"Split sizes | train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    train_csv, val_csv, test_csv = save_runtime_csvs(
        train_df, val_df, test_df, runtime_dir
    )

    generate_runtime_features(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        audio_dir=args.audio_dir,
        openface_dir=args.openface_dir,
        feature_dir=feature_dir
    )

    visual_train = os.path.join(feature_dir, "visual_train.npy")
    visual_val   = os.path.join(feature_dir, "visual_val.npy")
    visual_test  = os.path.join(feature_dir, "visual_test.npy")

    acoustic_train = os.path.join(feature_dir, "acoustic_train.npy")
    acoustic_val   = os.path.join(feature_dir, "acoustic_val.npy")
    acoustic_test  = os.path.join(feature_dir, "acoustic_test.npy")

    X_train = load_features(args.feature_type, visual_train, acoustic_train)
    X_val   = load_features(args.feature_type, visual_val, acoustic_val)
    X_test  = load_features(args.feature_type, visual_test, acoustic_test)

    inspect_array("X_train", X_train)
    inspect_array("X_val", X_val)
    inspect_array("X_test", X_test)

    y_train = load_labels(train_csv, args.task)
    y_val   = load_labels(val_csv, args.task)
    y_test  = load_labels(test_csv, args.task)

    print("Feature shapes:")
    print("  X_train:", X_train.shape)
    print("  X_val  :", X_val.shape)
    print("  X_test :", X_test.shape)

    class_weight = "balanced" if args.imbalanced == 1 else None

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=args.C,
            class_weight=class_weight,
            random_state=args.seed,
            max_iter=args.max_iter
        )
    )

    start_time = time.time()
    clf.fit(X_train, y_train)
    elapsed_min = (time.time() - start_time) / 60.0

    train_pred = clf.predict(X_train)
    val_pred   = clf.predict(X_val)
    test_pred  = clf.predict(X_test)

    train_result = evaluate(y_train, train_pred)
    val_result   = evaluate(y_val, val_pred)
    test_result  = evaluate(y_test, test_pred)

    save_dir = os.path.join(
        "output",
        "svm_results",
        f"{args.task}_seed{args.seed}_fold{args.fold_idx}"
    )
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "task": args.task,
        "seed": args.seed,
        "fold_idx": args.fold_idx,
        "feature_type": args.feature_type,
        "imbalanced": args.imbalanced,
        "C": args.C,
        "max_iter": args.max_iter,
        "train": train_result,
        "val": val_result,
        "test": test_result,
        "elapsed_min": round(elapsed_min, 3),
        "timestamp": datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    }

    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\nTrain result:")
    print(json.dumps(train_result, indent=2))
    print("\nVal result:")
    print(json.dumps(val_result, indent=2))
    print("\nTest result:")
    print(json.dumps(test_result, indent=2))


if __name__ == "__main__":
    main()