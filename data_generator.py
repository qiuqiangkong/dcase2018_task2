import h5py
import numpy as np
<<<<<<< HEAD
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
=======
import matplotlib.pyplot as plt
import time

from utilities import scale, pad_seq, mat_2d_to_3d, pad_or_trunc
import config as config

class Generator(object):
    def __init__(self, hdf5_path, batch_size, time_steps, verified_only=True, validation=True, seed=1234):
        
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.random_state = np.random.RandomState(seed)
        
        load_time = time.time()
        
        with h5py.File(hdf5_path, 'r') as hf:
            
            self.filenames = [name.decode() for name in hf['filenames'][:]]
            self.x = hf['data'][:]
            labels = hf['labels'][:]
            manually_verifications = hf['manually_verifications'][:]
            validations = hf['validations'][:]
            self.bgn_fin_indices = hf['bgn_fin_indices'][:]
            mean = hf['scalar']['mean'][:]
            std = hf['scalar']['std'][:]
            
        print("Load data time: {}".format(time.time() - load_time))

        self.x = scale(self.x, mean, std)
        self.y = np.array([config.lb_to_ix[lb.decode()] for lb in labels])
        
        num_audios = len(self.bgn_fin_indices)
        audio_ids = np.arange(num_audios)
        
        train_ids = []
        valid_ids = []
        
        for n in range(num_audios):
            
            selected_bool = (not verified_only) or (manually_verifications[n] == 1)
            valid_bool = validation and (validations[n] == 1)
            
            if selected_bool and (not valid_bool):
                train_ids.append(n)
                
            if selected_bool and valid_bool:
                valid_ids.append(n)
                
        self.train_ids = np.array(train_ids)
        self.valid_ids = np.array(valid_ids)
>>>>>>> 629f29cec5fea0c2044b910e14aa6c64291e3600
        
    def generate(self):
        
        batch_size = self.batch_size
<<<<<<< HEAD
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
=======
        train_ids = self.train_ids
        time_steps = self.time_steps
        
        num_train_audios = len(train_ids)
>>>>>>> 629f29cec5fea0c2044b910e14aa6c64291e3600
        
        iteration = 0
        pointer = 0
        
<<<<<<< HEAD
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
=======
        self.random_state.shuffle(train_ids)
        
        while True:
            
            # Reset pointer
            if pointer >= num_train_audios:
                
                pointer = 0
                self.random_state.shuffle(train_ids)
            
            batch_train_ids = train_ids[pointer : pointer + batch_size]
            pointer += batch_size
            
            batch_bgn_fin_indices = self.bgn_fin_indices[batch_train_ids]
            
            iteration += 1
            
            batch_x = []
            
            for m in range(len(batch_bgn_fin_indices)):
                [bgn, fin] = batch_bgn_fin_indices[m]
            
                if fin - bgn <= time_steps:
                    slice_x = self.x[bgn : fin]
                    slice_x = pad_seq(slice_x, time_steps)
                    
                else:
                    new_bgn = self.random_state.randint(low=bgn, high=fin-time_steps, size=1)[0]
                    
                    slice_x = self.x[new_bgn : new_bgn + time_steps]
                    
                    # plt.matshow(slice_x.T, origin='lower', aspect='auto', cmap='jet')
                    # plt.show()
            
                batch_x.append(slice_x)
                
            batch_x = np.array(batch_x)
            
            batch_y = self.y[batch_train_ids]
            
            yield batch_x, batch_y
        
    def generate_train_slices(self, max_audios=None):
        return self.generate_slices(self.train_ids, max_audios)
        
        
    def generate_validate_slices(self, max_audios=None):
        return self.generate_slices(self.valid_ids, max_audios)
        
        
    def generate_slices(self, ids, max_audios=None):

        count = 0
        
        for id in ids:
>>>>>>> 629f29cec5fea0c2044b910e14aa6c64291e3600
            
            if count == max_audios:
                break
                
<<<<<<< HEAD
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
=======
            [bgn, fin] = self.bgn_fin_indices[id]
            slice_x = self.x[bgn : fin]
            slice_y = self.y[id]
            filename = self.filenames[id]
            
            count += 1
            
            yield slice_x, slice_y, filename
            
            
class TestGenerator(object):
    def __init__(self, tr_hdf5_path, te_hdf5_path):
        
        with h5py.File(tr_hdf5_path, 'r') as hf:
            mean = hf['scalar']['mean'][:]
            std = hf['scalar']['std'][:]
            
        with h5py.File(te_hdf5_path, 'r') as hf:
            self.x = hf['data'][:]
            self.audio_names = hf['filenames'][:]
            self.bgn_fin_indices = hf['bgn_fin_indices'][:]
            
        self.x = scale(self.x, mean, std)
        self.audio_names = [audio_name.decode() for audio_name in self.audio_names]
        
    def generate_test_slices(self):
        
        num_audios = len(self.bgn_fin_indices)
        
        for n in range(num_audios):
            [bgn, fin] = self.bgn_fin_indices[n]
            slice_x = self.x[bgn : fin]
            audio_name = self.audio_names[n]
            yield slice_x, audio_name
            
            
class GeneratorValidationTwo(object):
    def __init__(self, hdf5_path, batch_size, time_steps, verified_only=True, validation=True, seed=1234):
        
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.random_state = np.random.RandomState(seed)
        
        load_time = time.time()
        
        with h5py.File(hdf5_path, 'r') as hf:
            
            self.filenames = [name.decode() for name in hf['filenames'][:]]
            self.x = hf['data'][:]
            labels = hf['labels'][:]
            manually_verifications = hf['manually_verifications'][:]
            validations = hf['validations'][:]
            self.bgn_fin_indices = hf['bgn_fin_indices'][:]
            mean = hf['scalar']['mean'][:]
            std = hf['scalar']['std'][:]
            
        print("Load data time: {}".format(time.time() - load_time))

        self.x = scale(self.x, mean, std)
        self.y = np.array([config.lb_to_ix[lb.decode()] for lb in labels])
        
        num_audios = len(self.bgn_fin_indices)
        
        train_ids = []
        valid_1_ids = []
        valid_2_ids = []
        
        for n in range(num_audios):
            
            selected_bool = (not verified_only) or (manually_verifications[n] == 1)
            valid_1_bool = validation and (validations[n] == 1)
            valid_2_bool = validation and (validations[n] == 2)
            train_bool = not (valid_1_bool or valid_2_bool)
            
            if selected_bool and train_bool:
                train_ids.append(n)
                
            if selected_bool and valid_1_bool:
                valid_1_ids.append(n)
                
            if selected_bool and valid_2_bool:
                valid_2_ids.append(n)
                
        self.train_ids = np.array(train_ids)
        self.valid_1_ids = np.array(valid_1_ids)
        self.valid_2_ids = np.array(valid_2_ids)
        
    def generate(self):
        
        batch_size = self.batch_size
        train_ids = self.train_ids
        time_steps = self.time_steps
        
        num_train_audios = len(train_ids)
        
        iteration = 0
        pointer = 0
        
        self.random_state.shuffle(train_ids)
        
        while True:
            
            # Reset pointer
            if pointer >= num_train_audios:
                
                pointer = 0
                self.random_state.shuffle(train_ids)
            
            batch_train_ids = train_ids[pointer : pointer + batch_size]
            pointer += batch_size
            
            batch_bgn_fin_indices = self.bgn_fin_indices[batch_train_ids]
            
            iteration += 1
            
            batch_x = []
            
            for m in range(len(batch_bgn_fin_indices)):
                [bgn, fin] = batch_bgn_fin_indices[m]
            
                if fin - bgn <= time_steps:
                    slice_x = self.x[bgn : fin]
                    slice_x = pad_seq(slice_x, time_steps)
                    
                else:
                    new_bgn = self.random_state.randint(low=bgn, high=fin-time_steps, size=1)[0]
                    
                    slice_x = self.x[new_bgn : new_bgn + time_steps]
                    
                batch_x.append(slice_x)
                
            batch_x = np.array(batch_x)
            
            batch_y = self.y[batch_train_ids]
            
            yield batch_x, batch_y
        
    def generate_train_slices(self, max_audios=None):
        return self.generate_slices(self.train_ids, max_audios)
        
    def generate_validate_slices(self, validation, max_audios=None):
        if validation == 1:
            valid_ids = self.valid_1_ids
        elif validation == 2:
            valid_ids = self.valid_2_ids
        else:
            raise Exception("Incorrect validation!")
            
        return self.generate_slices(valid_ids, max_audios)
        
    def generate_slices(self, ids, max_audios=None):

        count = 0
        
        for id in ids:
            
            if count == max_audios:
                break
                
            [bgn, fin] = self.bgn_fin_indices[id]
            slice_x = self.x[bgn : fin]
            slice_y = self.y[id]
            filename = self.filenames[id]
            
            count += 1
            
            yield slice_x, slice_y, filename
 
>>>>>>> 629f29cec5fea0c2044b910e14aa6c64291e3600
