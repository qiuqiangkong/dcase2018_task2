#!/bin/bash
<<<<<<< HEAD
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
=======
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task2"
WORKSPACE="/vol/vssp/msos/qk/workspaces/dcase2018_task2"

# Create validation csv
python create_validation.py validation_two --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Calculate features
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train model
CUDA_VISIBLE_DEVICES=4 python main_cnn_validation_two.py train --workspace=$WORKSPACE --verified_only=False --validation=True

# Inference on validation
CUDA_VISIBLE_DEVICES=4 python main_cnn_validation_two.py inference_validation --workspace=$WORKSPACE --iteration=3000


######
# Train model on full data
CUDA_VISIBLE_DEVICES=4 python main_cnn_validation_two.py train --workspace=$WORKSPACE --verified_only=False --validation=False

# Inference on private data
CUDA_VISIBLE_DEVICES=4 python main_cnn_validation_two.py inference_private --workspace=$WORKSPACE --iteration=3000

>>>>>>> 629f29cec5fea0c2044b910e14aa6c64291e3600
