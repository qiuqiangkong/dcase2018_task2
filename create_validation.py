import numpy as np
import os
import pandas as pd
import argparse
import random

import config as cfg


def create_validation(args):
    """Read train.csv and add a validation flag, then write out to 
    validation.csv
    """

    random.seed(1234)
    
    validation_audios_per_class = 20
    """Total number for validation is 20 * 41 (classes) = 820"""
    
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    
    labels = cfg.labels
    
    csv_path = os.path.join(dataset_dir, 'train.csv')
    
    df = pd.DataFrame(pd.read_csv(csv_path))
    
    dict = {label: [] for label in labels}
    
    num_rows = df.shape[0]
    
    # Find out varified audios
    for n in range(num_rows):
        fname = df.iloc[n]['fname']
        label = df.iloc[n]['label']
        manually_verified = df.iloc[n]['manually_verified']
    
        if manually_verified == 1:
            dict[label].append(fname)
    
    # Keep audios for validation
    validation_names = []
    
    for label in labels:
        random.shuffle(dict[label])
        validation_names += dict[label][0 : validation_audios_per_class]

    # Write out validation csv
    df_ex = df
    df_ex['validation'] = 0
    
    for n in range(num_rows):
        fname = df_ex.iloc[n]['fname']
        
        if fname in validation_names:
            
            df_ex.iloc[n, 3] = 1
            
    out_path = os.path.join(workspace, 'validation.csv')
    df_ex.to_csv(out_path)
    print("Write out to {}".format(out_path))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--workspace')
    
    args = parser.parse_args()
    
    create_validation(args)