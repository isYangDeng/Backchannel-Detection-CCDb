#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import subprocess
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run 11-fold cross validation")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_type", type=str, default="combined",
                        choices=["visual", "acoustic", "combined"])
    parser.add_argument("--imbalanced", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    for fold_idx in range(11):
        cmd = [
            "python", "model/main.py",
            "--task", args.task,
            "--seed", str(args.seed),
            "--fold_idx", str(fold_idx),
            "--feature_type", args.feature_type,
            "--imbalanced", str(args.imbalanced),
            "--train_mode", "True",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    metrics = []
    for fold_idx in range(11):
        result_path = os.path.join(
            "output",
            "checkpoints",
            f"{args.task}_seed{args.seed}_fold{fold_idx}",
            "result.json"
        )
        with open(result_path, "r") as f:
            result = json.load(f)
        metrics.append(result["test"])

    print("\n===== 11-fold summary =====")
    for key in ["accuracy", "precision", "recall", "f1"]:
        values = [m[key] for m in metrics]
        print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    main()
'''
11 fold
nohup python model/run_cv.py --task Backchannel --seed 32678 --feature_type combined > bc_transformer_combined_seed32678.log 2>&1 & 
nohup python model/run_cv.py --task Backchannel --seed 1234 --feature_type combined > bc_transformer_combined_seed1234.log 2>&1 &
nohup python model/run_cv.py --task Backchannel --seed 734783 --feature_type visual > bc_transformer_visual_seed734783.log 2>&1 &
nohup python model/run_cv.py --task Backchannel --seed 4234234 --feature_type acoustic > bc_transformer_acoustic_seed4234234.log 2>&1 &



nohup python model/run_cv.py --task Backchannel --seed 42 --feature_type combined > logs/bc_transformer_seed42.log 2>&1 & 
nohup python model/run_cv.py --task Backchannel --seed 10000 --feature_type combined > logs/bc_transformer_seed10000.log 2>&1 & 
nohup python model/run_cv.py --task Backchannel --seed 1111 --feature_type combined > logs/bc_transformer_seed1111.log 2>&1 & 
nohup python model/run_cv.py --task Backchannel --seed 11 --feature_type combined > logs/bc_transformer_seed11.log 2>&1 & 
'''