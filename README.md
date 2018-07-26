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
**Prepare data.** Download and upzip the data. The data looks like:

<pre>
.
├── audio_train (9473 audios)
│     └── ...
├── audio_test (9400 audios)
│     └── ...
├── train.csv
└── sample_submission.csv
</pre>

**1. (Optional) Install dependent packages.** The code is implememnted with python 3. If you are using conda, simply run:

$ conda env create -f environment.yml

$ conda activate py3_dcase2018_task2

**2. Then simply run:**

$ ./runme.sh

Or run the commands in runme.sh line by line, including: 

(1) Modify the paths of data and your workspace

(2) Extract features

(3) Train model

(4) Evaluation

## Result

We apply a convolutional neural network on the log mel spectrogram feature to solve this task. Training takes around 300 ms / iteration on a GTX Titan X GPU for the VGGish model. The training almost converge when after 3000 iterations. 

<pre>
Loading data time: 1.708 s
Training audios number: 7104
Validation audios number: 2369
Training patches number: 18777
train acc: 0.020, train mapk: 0.032
valid acc: 0.024, validate mapk: 0.040
------------------------------------
Iteration: 0, train time: 0.004 s, eval time: 2.829 s
train acc: 0.747, train mapk: 0.828
valid acc: 0.693, validate mapk: 0.775
------------------------------------
......
------------------------------------
Iteration: 2800, train time: 59.517 s, eval time: 3.286 s
train acc: 1.000, train mapk: 1.000
valid acc: 0.897, validate mapk: 0.930
......
</pre>

### Overall accuracy

We split development data to 4 folds. The overall performance on the 4 folds is:

|       | accuracy | mAP@3 |
|:-----:|----------|-------|
| Total | 0.895    | 0.928 |


### Class-wise accuracy

The class-wise accuracy is shown as blow:

![alt text](appendixes/class_wise_accuracy.png)

## Summary
This codebase provides a convolutional neural network (CNN) for DCASE 2018 challenge Task 2. Some sound classes such as 'applause', 'bark' have high classification accuracy. Some sound classes such as 'squeak', 'telephone' have low classification accuracy. 

### External link

The official baseline system implemented using Keras can be found in https://github.com/DCASE-REPO/dcase2018_baseline
