import h5py
import numpy as np
import time
import pandas as pd
import logging
import matplotlib.pyplot as plt

from utilities import calculate_scalar, repeat_seq, scale
import config


class DataGenerator(object):
    
    def __init__(self, hdf5_path, batch_size, verified_only, 
        validation_csv=None, seed=1234):
        """
        Inputs:
          hdf5_path: string
          batch_size: int
          verified_only: bool
          validate_csv: string | None, if None then use all data for training
          seed: int, random seed
        """
        
        # Parameters
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        
        lb_to_ix = config.lb_to_ix
        self.time_steps = config.time_steps
        self.hop_frames = self.time_steps // 2
        
        # Load data
        load_time = time.time()
        hf = h5py.File(hdf5_path, 'r')
            
        self.audio_names = np.array([s.decode() for s in hf['filename']])
        self.x = hf['feature'][:]
        self.bgn_fin_indices = hf['bgn_fin_indices'][:]
        target_labels = hf['label'][:]
        self.manually_verifications = hf['manually_verification'][:]
        
        self.y = np.array([lb_to_ix[s.decode()] for s in target_labels])
        
        hf.close()
        
        logging.info("Loading data time: {:.3f} s".format(
            time.time() - load_time))
        
        # Load validation
        if validation_csv:
            self.validations = self.calculate_validations(self.audio_names, 
                                                          validation_csv)
        
        # Calculate scalar
        scalar_time = time.time()
        
        (self.mean, self.std) = calculate_scalar(self.x)
        
        logging.info("Calculating scalar time: {:.3f} s".format(
            time.time() - scalar_time))

        # Get train & validate audio indexes
        audios_num = len(self.bgn_fin_indices)
        
        train_audio_indexes = []
        valid_audio_indexes = []
        
        for n in range(audios_num):
            
            selected_bool = (not verified_only) or \
                (self.manually_verifications[n] == 1)
                
            validate_bool = validation_csv and (self.validations[n] == 1)
            
            if selected_bool:
                
                if validate_bool:
                    valid_audio_indexes.append(n)
                
                else:
                    train_audio_indexes.append(n)

        self.train_audio_indexes = np.array(train_audio_indexes)
        self.valid_audio_indexes = np.array(valid_audio_indexes)
        
        logging.info("Training audios number: {}".format(
            len(self.train_audio_indexes)))
            
        logging.info("Validation audios number: {}".format(
            len(self.valid_audio_indexes)))
        
    def calculate_validations(self, audio_names, validation_csv):
        """Load validation information from validation csv. 
        """
        
        df = pd.read_csv(validation_csv, sep=',')
        df = pd.DataFrame(df)
        
        validations = []
        
        for audio_name in audio_names:
            
            row = df.query("fname == '{}'".format(audio_name))
            validation = row['validation'].values[0]
            validations.append(validation)
            
        validations = np.array(validations)
        
        return validations

        
    def generate(self):
        
        batch_size = self.batch_size
        audio_indexes = self.train_audio_indexes
        time_steps = self.time_steps
        hop_frames = self.hop_frames

        # Obtain training patches and corresponding targets
        bgn_fins = []
        ys = []
        
        for n in range(len(audio_indexes)):
            
            [bgn, fin] = self.bgn_fin_indices[audio_indexes][n]
            slice_y = self.y[audio_indexes][n]
        
            pointer = bgn
            
            while pointer + time_steps < fin:
                bgn_fins.append([pointer, pointer + time_steps])
                pointer += hop_frames
                ys.append(slice_y)
                
            bgn_fins.append([pointer, fin])
            ys.append(slice_y)
        
        bgn_fins = np.array(bgn_fins)
        ys = np.array(ys)
        
        logging.info("Number of training patches: {}".format(len(ys)))

        
        iteration = 0
        pointer = 0
        
        patch_indexes = np.arange(len(ys))
        
        self.random_state.shuffle(patch_indexes)
        
        # Generate mini-batch
        while True:
            
            # Reset pointer
            if pointer >= len(ys):
                
                pointer = 0
                self.random_state.shuffle(patch_indexes)
            
            batch_patch_indexes = patch_indexes[pointer : pointer + batch_size]
            pointer += batch_size
            
            iteration += 1
            
            batch_bgn_fins = bgn_fins[batch_patch_indexes]
            batch_y = ys[batch_patch_indexes]
            
            batch_x = []
            
            for n in range(len(batch_bgn_fins)):
                
                [bgn, fin] = batch_bgn_fins[n]
                patch_x = self.x[bgn : fin]
                
                if len(patch_x) < time_steps:
                    patch_x = repeat_seq(patch_x, time_steps)
                
                patch_x = self.transform(patch_x)
                
                batch_x.append(patch_x)
                
            batch_x = np.array(batch_x)
            
            yield batch_x, batch_y
        
    def generate_train_slices(self, hop_frames, max_audios=None):

        return self.generate_slices(audio_indexes=self.train_audio_indexes, 
                                    hop_frames=hop_frames, 
                                    max_audios=max_audios)
        
    def generate_validate_slices(self, hop_frames, max_audios=None):
        
        return self.generate_slices(audio_indexes=self.valid_audio_indexes, 
                                    hop_frames=hop_frames, 
                                    max_audios=max_audios)
        
    def generate_slices(self, audio_indexes, hop_frames, max_audios=None):

        time_steps = self.time_steps

        count = 0
        
        for audio_index in audio_indexes:

            
            if count == max_audios:
                break

            [bgn, fin] = self.bgn_fin_indices[audio_index]
            slice_x = self.x[bgn : fin]

            if len(slice_x) < time_steps:
                slice_x = repeat_seq(slice_x, time_steps)
            
            pointer = 0
            batch_x = []
            batch_y = []
            
            while(pointer + time_steps <= len(slice_x)):
                
                patch_x = slice_x[pointer : pointer + time_steps]
                
                patch_x = self.transform(patch_x)
                
                pointer += hop_frames
                
                batch_x.append(patch_x)
                batch_y.append(self.y[audio_index])

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            audio_name = self.audio_names[audio_index]

            count += 1
            
            yield batch_x, batch_y, audio_name
            
    def transform(self, x):
        """Transform data. 
        
        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)
          
        Returns:
          Transformed data. 
        """

        return scale(x, self.mean, self.std)
            
            
class TestDataGenerator(DataGenerator):
    
    def __init__(self, dev_hdf5_path, test_hdf5_path, test_hop_frames):
        
        super(TestDataGenerator, self).__init__(
            hdf5_path=dev_hdf5_path, 
            batch_size=None, 
            verified_only=False,
            validation_csv=None)
        
        self.test_hop_frames = test_hop_frames
        
        # Load test data
        load_time = time.time()
        hf = h5py.File(test_hdf5_path, 'r')

        self.test_audio_names = np.array([s.decode() for s in hf['filename'][:]])
        self.test_x = hf['feature'][:]
        self.test_bgn_fin_indices = hf['bgn_fin_indices'][:]
        
        hf.close()
        
        logging.info("Loading data time: {:.3f} s".format(
            time.time() - load_time))
        
    def generate_test_slices(self):
        
        time_steps = self.time_steps
        test_hop_frames = self.test_hop_frames
        corrupted_files = config.corrupted_files
        
        audios_num = len(self.test_audio_names)
        
        for n in range(audios_num):
            
            audio_name = self.test_audio_names[n]
            
            if audio_name in corrupted_files:
                logging.info("File {} is corrupted!".format(audio_name))
                
            else:
                [bgn, fin] = self.test_bgn_fin_indices[n]
                
                slice_x = self.test_x[bgn : fin]
    
                if len(slice_x) < time_steps:
                    slice_x = repeat_seq(slice_x, time_steps)
                
                pointer = 0
                batch_x = []
                
                while(pointer + time_steps <= len(slice_x)):
                    
                    patch_x = slice_x[pointer : pointer + time_steps]
                    
                    patch_x = self.transform(patch_x)
                    
                    pointer += test_hop_frames
                    
                    batch_x.append(patch_x)
    
                batch_x = np.array(batch_x)
                
                yield batch_x, audio_name