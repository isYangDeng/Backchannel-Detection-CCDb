#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
import subprocess
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run 11-fold CV for one task using SVM")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_type", type=str, default="combined",
                        choices=["visual", "acoustic", "combined"])
    parser.add_argument("--imbalanced", type=int, default=0)
    parser.add_argument("--root", type=str, default="output/svm_results")
    parser.add_argument("--summary_split", type=str, default="test",
                        choices=["train", "val", "test"])
    return parser.parse_args()


def summarize_results(task, seed, root, split="test"):
    pattern = os.path.join(root, f"{task}_seed{seed}_fold*", "result.json")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No result files found: {pattern}")

    accs = []
    precisions = []
    recalls = []
    f1s = []

    print(f"\nFound {len(files)} folds for summary\n")

    for fp in files:
        with open(fp, "r") as f:
            data = json.load(f)

        fold_idx = data["fold_idx"]
        metrics = data[split]

        acc = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(
            f"fold {fold_idx:02d} | "
            f"accuracy={acc:.6f} | "
            f"precision={precision:.6f} | "
            f"recall={recall:.6f} | "
            f"f1={f1:.6f}"
        )

    accs = np.array(accs, dtype=float)
    precisions = np.array(precisions, dtype=float)
    recalls = np.array(recalls, dtype=float)
    f1s = np.array(f1s, dtype=float)

    print("\n==============================")
    print(f"Task : {task}")
    print(f"Seed : {seed}")
    print(f"Split: {split}")
    print("==============================")
    print(f"Accuracy  mean ± std: {accs.mean():.6f} ± {accs.std(ddof=1):.6f}")
    print(f"Precision mean ± std: {precisions.mean():.6f} ± {precisions.std(ddof=1):.6f}")
    print(f"Recall    mean ± std: {recalls.mean():.6f} ± {recalls.std(ddof=1):.6f}")
    print(f"F1        mean ± std: {f1s.mean():.6f} ± {f1s.std(ddof=1):.6f}")


def main():
    args = parse_args()

    for fold_idx in range(11):
        print("")
        print("======================================")
        print(f"Running fold {fold_idx}/10")
        print("======================================")

        cmd = [
            "python", "src/main_svm.py",
            "--task", args.task,
            "--seed", str(args.seed),
            "--fold_idx", str(fold_idx),
            "--feature_type", args.feature_type,
            # "--imbalanced", str(args.imbalanced),
        ]

        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    summarize_results(
        task=args.task,
        seed=args.seed,
        root=args.root,
        split=args.summary_split
    )


if __name__ == "__main__":
    main()

'''
python src/run_cv_svm.py \
  --task Backchannel \
  --seed 42333333 \
  --feature_type combined
'''