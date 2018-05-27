import h5py
import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import scale, pad_seq
import config as config

class Generator(object):
    def __init__(self, hdf5_path, batch_size, time_steps, verified_only=True, validation=True, seed=1234):
        
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.random_state = np.random.RandomState(seed)
        
        load_time = time.time()
        
        with h5py.File(hdf5_path, 'r') as hf:
            
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
            
            if count == max_audios:
                break
                
            [bgn, fin] = self.bgn_fin_indices[id]
            slice_x = self.x[bgn : fin]
            slice_y = self.y[id]
            
            count += 1
            
            yield slice_x, slice_y
            
            
class Generator2(Generator):
    def __init__(self, hdf5_path, batch_size, time_steps, hop_frames, verified_only=True, validation=True, seed=1234):
        Generator.__init__(self, hdf5_path, batch_size, time_steps, verified_only, validation, seed)
        self.hop_frames = hop_frames
        
        self.train_bgn_fin_y_mat = self.get_bgn_fin_y_mat(self.train_ids)
        self.valid_bgn_fin_y_mat = self.get_bgn_fin_y_mat(self.train_ids)
        
    def get_bgn_fin_y_mat(self, ids):
        
        time_steps = self.time_steps
        hop_frames = self.hop_frames
        bgn_indexes = []
        fin_indexes = []
        ys = []
        
        for id in ids:
            (bgn, fin) = self.bgn_fin_indices[id]
            
            if fin - bgn < time_steps:
                bgn_indexes += [bgn]
                fin_indexes += [fin]
                ys += [self.y[id]]
                
            else:
                bgn_indexes += range(bgn, fin - time_steps, hop_frames)
                fin_indexes += range(bgn + time_steps, fin, hop_frames)
                ys += [self.y[id]] * len(range(bgn, fin - time_steps, hop_frames))
        
        bgn_fin_y_mat = np.array([bgn_indexes, fin_indexes, ys]).T
        
        return bgn_fin_y_mat
        
    def generate(self):
        
        batch_size = self.batch_size
        time_steps = self.time_steps
        bgn_fin_y_mat = self.train_bgn_fin_y_mat
        
        iteration = 0
        pointer = 0
        
        self.random_state.shuffle(bgn_fin_y_mat)
        
        while True:
            
            # Reset pointer
            if pointer >= len(bgn_fin_y_mat):
                
                pointer = 0
                self.random_state.shuffle(bgn_fin_y_mat)
            
            batch_bgn_fin_y = bgn_fin_y_mat[pointer : pointer + batch_size]
            pointer += batch_size
            
            iteration += 1
            
            batch_x = []
            batch_y = []
            
            for n in range(len(batch_bgn_fin_y)):
                (bgn, fin, slice_y) = batch_bgn_fin_y[n]
                
                if fin - bgn < time_steps:
                    slice_x = pad_seq(self.x[bgn : fin], time_steps)
                    
                else:
                    slice_x = self.x[bgn : fin]
                    
                if False:
                    plt.matshow(slice_x.T, origin='lower', aspect='auto', cmap='jet')
                    plt.show()
            
                batch_x.append(slice_x)
                batch_y.append(slice_y)
            
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            
            yield batch_x, batch_y
        
            
class TestGenerator(object):
    def __init__(self, tr_hdf5_path, te_hdf5_path):
        
        with h5py.File(tr_hdf5_path, 'r') as hf:
            mean = hf['train']['scalar']['mean'][:]
            std = hf['train']['scalar']['std'][:]
            
        with h5py.File(te_hdf5_path, 'r') as hf:
            self.x = hf['test']['data'][:]
            self.audio_names = hf['test']['filenames'][:]
            self.bgn_fin_indices = hf['test']['bgn_fin_indices'][:]
            
        self.x = scale(self.x, mean, std)
        self.audio_names = [audio_name.decode() for audio_name in self.audio_names]
        
    def generate_test_slices(self):
        
        num_audios = len(self.bgn_fin_indices)
        
        for n in range(num_audios):
            [bgn, fin] = self.bgn_fin_indices[n]
            slice_x = self.x[bgn : fin]
            audio_name = self.audio_names[n]
            yield slice_x, audio_name