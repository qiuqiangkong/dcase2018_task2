#!/bin/bash
# Data directory
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task2"

# Workspace
WORKSPACE="/vol/vssp/msos/qk/workspaces/pub_dcase2018_task2"

# Create validation csv
python create_validation.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Calculate features
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type=development
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type=test

############ Development ############
# Train
CUDA_VISIBLE_DEVICES=0 python main_pytorch.py train --workspace=$WORKSPACE --validate --cuda

# Validation
CUDA_VISIBLE_DEVICES=0 python main_pytorch.py inference_validation --workspace=$WORKSPACE --iteration=3000 --cuda

############ Full train ############
# Train on full data
CUDA_VISIBLE_DEVICES=0 python main_pytorch.py train --workspace=$WORKSPACE --cuda

# Inference testing data
CUDA_VISIBLE_DEVICES=0 python main_pytorch.py inference_testing_data --workspace=$WORKSPACE --iteration=3000 --cuda