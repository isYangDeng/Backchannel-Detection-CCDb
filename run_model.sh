#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  bash run.sh --model transformer --mode single --task TASK --fold_idx N [options]
  bash run.sh --model transformer --mode cv     --task TASK [options]
  bash run.sh --model svm         --mode single --task TASK --fold_idx N [options]
  bash run.sh --model svm         --mode cv     --task TASK [options]

Required arguments:
  --model         transformer | svm
  --mode          single | cv
  --task          task name, e.g. Backchannel
  --fold_idx      required only when --mode single

Common optional arguments:
  --seed          random seed (default: 42)
  --feature_type  visual | acoustic | combined (default: combined)
  --imbalanced    0 | 1 (default: 0)

Transformer-only optional arguments:
  --train_mode    True | False (default: True)
  --model_path    checkpoint path, required when train_mode=False
  --label_csv     default: output/label.csv
  --audio_dir     default: data/audio
  --openface_dir  default: data/openface_features
  --runtime_root  default: data/runtime
  --feature_root  default: data/features
  --batch_size    default: 64
  --num_epochs    default: 40
  --learning_rate default: 5e-5
  --weight_decay  default: 5e-4

SVM-only optional arguments:
  --label_csv     default: output/label.csv
  --audio_dir     default: data/audio
  --openface_dir  default: data/openface_features
  --runtime_root  default: data/runtime
  --feature_root  default: data/features
  --C             default: 1.0
  --max_iter      default: 5000

run_cv_svm-only optional arguments:
  --root          default: output/svm_results
  --summary_split train | val | test (default: test)

Examples:
    # Transformer (single fold)
    bash run_model.sh --model transformer --mode single --task Backchannel --fold_idx 0 --seed 1 --feature_type combined
    # Transformer (11-fold CV)
    bash run_model.sh --model transformer --mode cv --task Backchannel --seed 1 --feature_type combined
    # SVM (single fold)
    bash run_model.sh --model svm --mode single --task Backchannel --fold_idx 0 --seed 1 --feature_type visual
    # SVM (11-fold CV)
    bash run_model.sh --model svm --mode cv --task Backchannel --seed 1 --feature_type acoustic
EOF
}

MODEL=""
MODE=""
TASK=""
FOLD_IDX=""
SEED="42"
FEATURE_TYPE="combined"
IMBALANCED="0"

TRAIN_MODE="True"
MODEL_PATH=""
LABEL_CSV="output/label.csv"
AUDIO_DIR="data/audio"
OPENFACE_DIR="data/openface_features"
RUNTIME_ROOT="data/runtime"
FEATURE_ROOT="data/features"
BATCH_SIZE="64"
NUM_EPOCHS="40"
LEARNING_RATE="5e-5"
WEIGHT_DECAY="5e-4"

C_VALUE="1.0"
MAX_ITER="5000"

ROOT="output/svm_results"
SUMMARY_SPLIT="test"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --fold_idx) FOLD_IDX="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --feature_type) FEATURE_TYPE="$2"; shift 2 ;;
    --imbalanced) IMBALANCED="$2"; shift 2 ;;

    --train_mode) TRAIN_MODE="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --label_csv) LABEL_CSV="$2"; shift 2 ;;
    --audio_dir) AUDIO_DIR="$2"; shift 2 ;;
    --openface_dir) OPENFACE_DIR="$2"; shift 2 ;;
    --runtime_root) RUNTIME_ROOT="$2"; shift 2 ;;
    --feature_root) FEATURE_ROOT="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;

    --C) C_VALUE="$2"; shift 2 ;;
    --max_iter) MAX_ITER="$2"; shift 2 ;;

    --root) ROOT="$2"; shift 2 ;;
    --summary_split) SUMMARY_SPLIT="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$MODE" || -z "$TASK" ]]; then
  echo "Error: --model, --mode, and --task are required."
  usage
  exit 1
fi

if [[ "$MODE" == "single" && -z "$FOLD_IDX" ]]; then
  echo "Error: --fold_idx is required when --mode single."
  exit 1
fi

if [[ "$MODEL" != "transformer" && "$MODEL" != "svm" ]]; then
  echo "Error: --model must be 'transformer' or 'svm'."
  exit 1
fi

if [[ "$MODE" != "single" && "$MODE" != "cv" ]]; then
  echo "Error: --mode must be 'single' or 'cv'."
  exit 1
fi

if [[ "$FEATURE_TYPE" != "visual" && "$FEATURE_TYPE" != "acoustic" && "$FEATURE_TYPE" != "combined" ]]; then
  echo "Error: --feature_type must be visual, acoustic, or combined."
  exit 1
fi

mkdir -p logs

if [[ "$MODEL" == "transformer" && "$MODE" == "single" ]]; then
  CMD=(
    python src/main.py
    --task "$TASK"
    --seed "$SEED"
    --fold_idx "$FOLD_IDX"
    --feature_type "$FEATURE_TYPE"
    --imbalanced "$IMBALANCED"
    --train_mode "$TRAIN_MODE"
    --label_csv "$LABEL_CSV"
    --audio_dir "$AUDIO_DIR"
    --openface_dir "$OPENFACE_DIR"
    --runtime_root "$RUNTIME_ROOT"
    --feature_root "$FEATURE_ROOT"
    --batch_size "$BATCH_SIZE"
    --num_epochs "$NUM_EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
  )

  if [[ "$TRAIN_MODE" == "False" ]]; then
    if [[ -z "$MODEL_PATH" ]]; then
      echo "Error: --model_path is required when --train_mode False."
      exit 1
    fi
    CMD+=(--model_path "$MODEL_PATH")
  fi

elif [[ "$MODEL" == "transformer" && "$MODE" == "cv" ]]; then
  CMD=(
    python src/run_cv.py
    --task "$TASK"
    --seed "$SEED"
    --feature_type "$FEATURE_TYPE"
    --imbalanced "$IMBALANCED"
  )

elif [[ "$MODEL" == "svm" && "$MODE" == "single" ]]; then
  CMD=(
    python src/main_svm.py
    --task "$TASK"
    --seed "$SEED"
    --fold_idx "$FOLD_IDX"
    --feature_type "$FEATURE_TYPE"
    --imbalanced "$IMBALANCED"
    --label_csv "$LABEL_CSV"
    --audio_dir "$AUDIO_DIR"
    --openface_dir "$OPENFACE_DIR"
    --runtime_root "$RUNTIME_ROOT"
    --feature_root "$FEATURE_ROOT"
    --C "$C_VALUE"
    --max_iter "$MAX_ITER"
  )

elif [[ "$MODEL" == "svm" && "$MODE" == "cv" ]]; then
  CMD=(
    python src/run_cv_svm.py
    --task "$TASK"
    --seed "$SEED"
    --feature_type "$FEATURE_TYPE"
    --imbalanced "$IMBALANCED"
    --root "$ROOT"
    --summary_split "$SUMMARY_SPLIT"
  )
fi

echo "=================================================="
echo "Running command:"
printf '%q ' "${CMD[@]}"
echo
echo "=================================================="

"${CMD[@]}"
