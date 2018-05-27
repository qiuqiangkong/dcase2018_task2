#!/bin/bash
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task2"
WORKSPACE="/vol/vssp/msos/qk/workspaces/dcase2018_task2"

# Create validation csv
python create_validation.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Calculate features
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train model
CUDA_VISIBLE_DEVICES=6 python tmp01.py train --workspace=$WORKSPACE --verified_only=False --validation=True

CUDA_VISIBLE_DEVICES=6 python tmp01.py inference --workspace=$WORKSPACE --iteration=10000

