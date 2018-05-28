#!/bin/bash
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task2"
WORKSPACE="/vol/vssp/msos/qk/workspaces/dcase2018_task2"

# Create validation csv
python create_validation.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Calculate features
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train model
CUDA_VISIBLE_DEVICES=4 python main_cnn.py train --workspace=$WORKSPACE --verified_only=False --validation=True

# Inference on validation
CUDA_VISIBLE_DEVICES=4 python main_cnn.py inference_validation --workspace=$WORKSPACE --iteration=3000


######
# Train model on full data
CUDA_VISIBLE_DEVICES=4 python main_cnn.py train --workspace=$WORKSPACE --verified_only=False --validation=True

# Inference on private data
CUDA_VISIBLE_DEVICES=4 python main_cnn.py inference_private --workspace=$WORKSPACE --iteration=3000

