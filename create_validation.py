import numpy as np
import os
import pandas as pd
import argparse
import random

import config


def create_validation(args):
    """Create validation file.
    Write out a new csv file for validation, with an extra validation flag to
    existing train.csv. Validation data are extracted from manually verified
    data.

    Training: 2890 manually verfieid + 5763 not manually verified.
    Validation: 820 manually verified.
    """

    dataset_dir = args.dataset_dir
    workspace = args.workspace

    labels = config.labels

    random.seed(1234)
    validation_audios_per_class = 20    # In total 820 audios for validation.

    # Open csv
    csv_path = os.path.join(dataset_dir, 'train.csv')
    df = pd.DataFrame(pd.read_csv(csv_path))

    num_audios = df.shape[0]

    dict = {label: [] for label in labels}
    """key: label, value: list of manually verified audio names. """

    # Find out varified audios
    for n in range(num_audios):

        fname = df.iloc[n]['fname']
        label = df.iloc[n]['label']
        manually_verified = df.iloc[n]['manually_verified']

        if manually_verified == 1:
            dict[label].append(fname)

    # Names of audios for validation
    validation_names = []

    for label in labels:

        random.shuffle(dict[label])
        validation_names += dict[label][0: validation_audios_per_class]

    # Write out validation csv
    df_ex = df
    df_ex['validation'] = 0

    for n in range(num_audios):

        fname = df_ex.iloc[n]['fname']

        if fname in validation_names:

            df_ex.iloc[n, 3] = 1

    out_path = os.path.join(workspace, 'validate_meta.csv')
    df_ex.to_csv(out_path)

    print("Write out to {}".format(out_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--workspace')

    args = parser.parse_args()

    create_validation(args)
