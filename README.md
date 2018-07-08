# DCASE 2018 Task 2 Audio Tagging of Freesound

DCASE 2018 Task 2 is a task to classifiy short audio clips to one of 41 audio classes such as 'fireworks', 'cello', 'barks' and so on. We provide a convolutional neural network (CNN) baseline system implemented with PyTorch in this code base. More details about this challenge can be found http://dcase.community/challenge2018/task-general-purpose-audio-tagging

## DATASET

The dataset is downloadable from http://dcase.community/challenge2018/task-acoustic-scene-classification. There are 41 audio classes. The duration of the audio samples ranges from 300ms to 30s. 


|                       | Training | Testing                         |
|:---------------------:|----------|---------------------------------|
|   Manually verified   | 3710     | ~1.6k                           |
| Not manually verified | 5763     | ~7.8k (Not used for evaluation) |
| Total                 | 9473     | ~9.4k                           |

## Run the code

Install dependent packages. If you are using conda, simply run:
$ conda env create -f environment.yml
$ conda activate py3_dcase2018_task2

Run the commands in runme.sh line by line, including: 
(1) Modify the paths of data and your workspace
(2) Extract features
(3) Train model
(4) Evaluation

## Result

We apply a convolutional neural network on the log mel spectrogram feature to solve this task. Training takes around 100 ms / iteration on a GTX Titan X GPU. The model is trained for 3000 iterations. The result is shown below. 



## Summary
This codebase provides a convolutional neural network (CNN) for DCASE 2018 challenge Task 2. 

### External link

The official baseline system implemented using Keras can be found https://github.com/DCASE-REPO/dcase2018_baseline
