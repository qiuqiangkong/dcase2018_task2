import os
import numpy
import argparse
import sys
import soundfile
import numpy as np
import librosa
import h5py
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import logging

import config
    
   
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
   
   
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na
   
   
def create_logging(log_dir, filemode):
    
    create_folder(log_dir)
    i1 = 0
    
    while os.path.isfile(os.path.join(log_dir, "%04d.log" % i1)):
        i1 += 1
        
    log_path = os.path.join(log_dir, "%04d.log" % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
   

def read_audio(path, target_fs=None):

    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs


def calculate_scalar(x):

    if x.ndim == 2:
        axis = 0
        
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def scale(x, mean, std):

    return (x - mean) / std


def inverse_scale(x, mean, std):

    return x * std + mean


def repeat_seq(x, time_steps):
    repeat_num = time_steps // len(x) + 1
    repeat_x = np.tile(x, (repeat_num, 1))[0 : time_steps]
    return repeat_x
    
    
def calculate_accuracy(output, target):
    acc = np.sum(output == target) / float(len(target))
    return acc
    
    
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def calculate_mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def print_class_wise_accuracy(output, target):
    """Print class wise accuracy."""
    
    labels = config.labels
    ix_to_lb = config.ix_to_lb
    
    correct_dict = {label: 0 for label in labels}
    total_dict = {label: 0 for label in labels}
    
    for n in range(len(target)):
        
        label = ix_to_lb[target[n]]
        total_dict[label] += 1
        
        if output[n] == target[n]:
            correct_dict[label] += 1
        
    accuracy_array = []
    
    logging.info("")        
    for label in labels:
        accuracy = correct_dict[label] / float(total_dict[label])
        accuracy_array.append(accuracy)
        logging.info("{:<30}{}/{}\t{:.2f}".format(label, correct_dict[label], total_dict[label], accuracy))
        
    accuracy_array = np.array(accuracy_array)
    
    return accuracy_array

    
def plot_class_wise_accuracy(accuracy_array):
    """Plot accuracy."""
    
    labels = config.labels
    classes_num = len(labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.bar(np.arange(classes_num), accuracy_array, alpha=0.5)
    ax.set_xlim(0, classes_num)
    ax.set_ylim(0., 1.)
    ax.xaxis.set_ticks(np.arange(classes_num))
    ax.xaxis.set_ticklabels(labels, rotation=90, fontsize='large')
    plt.tight_layout()
    plt.show()