import os
import numpy as np
import soundfile
import librosa
import logging


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
    
    
def pad_seq(x, time_steps):
    (seq_len, feature_dim) = x.shape
    return np.concatenate((x, np.zeros((time_steps - seq_len, feature_dim))))


def pad_or_trunc(x, max_len):
    if len(x) == max_len:
        return x
    
    elif len(x) > max_len:
        return x[0 : max_len]
        
    else:
        (seq_len, freq_bins) = x.shape
        pad = np.zeros((max_len - seq_len, freq_bins))
        return np.concatenate((x, pad), axis=0)


def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    
    Args:
      x: 2darray, (n_time, n_in)
      agg_num: int, number of frames to concatenate. 
      hop: int, number of hop frames. 
      
    Returns:
      3darray, (n_blocks, agg_num, n_in)
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x + hop):
        x3d.append(pad_or_trunc(x[i1 : i1 + agg_num], agg_num))
        i1 += hop
    return np.array(x3d)