import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import time
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math
import h5py
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from utilities import (create_folder, get_filename, create_logging, 
                       calculate_accuracy, calculate_mapk, 
                       print_class_wise_accuracy, plot_class_wise_accuracy)
from models_pytorch import move_data_to_gpu, BaselineCnn
from data_generator import DataGenerator, TestDataGenerator
import config


kmax = config.kmax
time_steps = config.time_steps
train_hop_frames = time_steps // 2
test_hop_frames = config.test_hop_frames
    
    
def aggregate(outputs):
    
    agg_outputs = []
    
    for output in outputs:
        agg_output = np.mean(output, axis=0)
        agg_outputs.append(agg_output)
    
    agg_outputs = np.array(agg_outputs)
    
    return agg_outputs
    
 
def evaluate(model, generator, data_type, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      cuda: bool.
      
    Returns:
      accuracy: float
      mapk: float
    """

    model.eval()
    
    if data_type == 'train':
        generate_func = generator.generate_train_slices(
            hop_frames=test_hop_frames, max_audios=100)
    
    elif data_type == 'validate':
        generate_func = generator.generate_validate_slices(
            hop_frames=train_hop_frames, max_audios=None)
    
    # Forward
    (outputs, targets, audio_names) = forward(model=model, 
                                              generate_func=generate_func, 
                                              cuda=cuda, 
                                              has_target=True)
    '''outputs: (audios_num, patches_num, classes_num), targets: (audios_num,)'''

    outputs = aggregate(outputs)
    '''(audios_num, classes_num)'''

    predictions = np.argmax(outputs, axis=-1)
    '''(audios_num,)'''
    
    sorted_indices = np.argsort(outputs, axis=-1)[:, ::-1][:, :kmax]
    '''(audios_num, kmax)'''

    # Accuracy
    accuracy = calculate_accuracy(predictions, targets)

    # mAP
    mapk = calculate_mapk(targets[:, np.newaxis], sorted_indices, k=kmax)

    return accuracy, mapk
    
 
def forward(model, generate_func, cuda, has_target):
    """Forward data to a model.
    
    Args:
      model: object.
      generate_func: generate function
      cuda: bool
      has_target: bool
      
    Returns:
      (outputs, targets, audio_names) | (outputs, audio_names)
    """

    model.eval()

    outputs = []
    targets = []
    audio_names = []

    # Evaluate on mini-batch
    for data in generate_func:
            
        if has_target:
            (batch_x, batch_y, batch_audio_names) = data
            targets.append(batch_y[0])
            
        else:
            (batch_x, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        batch_output = model(batch_x)
        batch_output = batch_output.data.cpu().numpy()

        # Aggregate
        # output = aggregate(batch_output)
        
        outputs.append(batch_output)
        audio_names.append(batch_audio_names)

    outputs = np.array(outputs)
    audio_names = np.array(audio_names)
    
    if has_target:
        targets = np.array(targets)
        return outputs, targets, audio_names
        
    else:
        return outputs, audio_names
    
    
def train(args):
    
    # Arguments & parameters
    workspace = args.workspace
    filename = args.filename
    verified_only = args.verified_only
    validate = args.validate
    cuda = args.cuda
    mini_data = args.mini_data
    
    num_classes = len(config.labels)
    batch_size = 128

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel',
                                 'mini_development.h5')        
                                 
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel',
                                 'development.h5')
    
    if validate:
        validation_csv = os.path.join(workspace, 'validate_meta.csv')
        
        model_dir = os.path.join(workspace, 'models', filename, 
            'verified_only={}, validate={}'.format(verified_only, validate))
        
    else:
        validation_csv = None
    

    
    create_folder(model_dir)
    
    # Model
    model = BaselineCnn(num_classes)
    
    if cuda:
        model.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), 
                           eps=1e-08, weight_decay=0.)
    
    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path, 
                              batch_size=batch_size, 
                              verified_only=verified_only, 
                              validation_csv=validation_csv)

    iteration = 0
    
    train_bgn_time = time.time()

    # Train on mini batches
    for (batch_x, batch_y) in generator.generate():
        
        # Evaluate
        if iteration % 100 == 0:

            train_fin_time = time.time()
            
            (tr_acc, tr_mapk) = evaluate(model=model, 
                                         generator=generator, 
                                         data_type='train', 
                                         cuda=cuda)
                                         
            logging.info("train acc: {:.3f}, train mapk: {:.3f}".format(
                tr_acc, tr_mapk))
                        
            if validate:
                (va_acc, va_mapk) = evaluate(model=model, 
                                             generator=generator, 
                                             data_type='validate', 
                                             cuda=cuda)
                                             
                logging.info("valid acc: {:.3f}, validate mapk: {:.3f}".format(
                    va_acc, va_mapk))
        
            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            
            logging.info("------------------------------------")
            logging.info("Iteration: {}, train time: {:.3f} s, eval time: {:.3f} s".format(
                iteration, train_time, validate_time))
            
            train_bgn_time = time.time()
        
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        
        # Forward
        t_forward = time.time()
        model.train()
        output = model(batch_x)
        
        # Loss
        loss = F.nll_loss(output, batch_y)
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iteration += 1
        
        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            save_out_dict = {'iteration': iteration, 
                             'state_dict': model.state_dict(), 
                             'optimizer': optimizer.state_dict(), }
            
            save_out_path = os.path.join(model_dir, 
                'md_{}_iters.tar'.format(iteration))
                
            torch.save(save_out_dict, save_out_path)
            logging.info("Save model to {}".format(save_out_path))
    
    
def inference_validation(args):
    
    # Arguments & parameters
    workspace = args.workspace
    verified_only = args.verified_only
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda
    
    num_classes = len(config.labels)
    
    # Paths
    model_path = os.path.join(workspace, 'models', filename, 
        'verified_only={}, validate=True'.format(verified_only), 
        'md_{}_iters.tar'.format(iteration))
    
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'development.h5')
    
    validation_csv = os.path.join(workspace, 'validate_meta.csv')
    
    # Model
    model = BaselineCnn(num_classes)
        
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()
    
    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path, 
                              batch_size=None, 
                              verified_only=verified_only, 
                              validation_csv=validation_csv)
    
    generate_func = generator.generate_validate_slices(
        hop_frames=test_hop_frames)
    
    # Forward
    (outputs, targets, audio_names) = forward(model=model, 
                                              generate_func=generate_func, 
                                              cuda=cuda, 
                                              has_target=True)
    '''outputs: (audios_num, patches_num, classes_num), targets: (audios_num,)'''

    outputs = aggregate(outputs)
    '''(audios_num, classes_num)'''

    predictions = np.argmax(outputs, axis=-1)
    '''(audios_num,)'''
    
    sorted_indices = np.argsort(outputs, axis=-1)[:, ::-1][:, :kmax]
    '''(audios_num, kmax)'''

    # Accuracy
    accuracy = calculate_accuracy(predictions, targets)

    # mAP
    mapk = calculate_mapk(targets[:, np.newaxis], sorted_indices, k=kmax)
    
    logging.info("")
    logging.info("iteration: {}".format(iteration))
    logging.info("accuracy: {:.3f}".format(accuracy))
    logging.info("mapk: {:.3f}".format(mapk))
        
    accuracy_array = print_class_wise_accuracy(predictions, targets)
    
    plot_class_wise_accuracy(accuracy_array)
    
    
def inference_testing_data(args):
    
    # Arguments & parameters
    workspace = args.workspace
    verified_only = args.verified_only
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda
    
    num_classes = len(config.labels)
    ix_to_lb = config.ix_to_lb
    
    # Paths
    model_path = os.path.join(workspace, 'models', filename, 
        'verified_only={}, validate=False'.format(verified_only), 
        'md_{}_iters.tar'.format(iteration))
    
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', 
                                 'development.h5')
                                 
    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', 'test.h5')
    
    submission_csv = os.path.join(workspace, 'submissions', filename, 
        'verified_only={}'.format(verified_only), 
        'iteration={}'.format(iteration), 'submission.csv')
    
    create_folder(os.path.dirname(submission_csv))
    
    # Model
    model = BaselineCnn(num_classes)
        
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()
    
    # Data generator
    test_generator = TestDataGenerator(dev_hdf5_path=dev_hdf5_path, 
                                       test_hdf5_path=test_hdf5_path, 
                                       test_hop_frames=test_hop_frames)
    
    generate_func = test_generator.generate_test_slices()
    
    # Forward
    (outputs, audio_names) = forward(model=model, 
                                     generate_func=generate_func, 
                                     cuda=cuda, 
                                     has_target=False)
    
    outputs = aggregate(outputs)
    '''(audios_num, classes_num)'''

    predictions = np.argmax(outputs, axis=-1)
    '''(audios_num,)'''
    
    sorted_indices = np.argsort(outputs, axis=-1)[:, ::-1][:, :kmax]
    '''(audios_num, kmax)'''
    
    # Write result to submission csv
    f = open(submission_csv, 'w')
    
    f.write('fname,label\n')
    
    for (n, audio_name) in enumerate(audio_names):
        
        f.write('{}, '.format(audio_name))
        
        for k in range(kmax):
            predicted_target = sorted_indices[n][k]
            predicted_label = ix_to_lb[predicted_target]

            f.write('{} '.format(predicted_label))
            
        f.write('\n')
    
    f.write('{},{}\n'.format('0b0427e2.wav', 'Acoustic_guitar'))
    f.write('{},{}\n'.format('6ea0099f.wav', 'Acoustic_guitar'))
    f.write('{},{}\n'.format('b39975f5.wav', 'Acoustic_guitar'))
        
    f.close()
    
    print("Write result to {}".format(submission_csv))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str)
    parser_train.add_argument('--verified_only', action='store_true', default=False)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--workspace', type=str)    
    parser_inference_validation.add_argument('--verified_only', action='store_true', default=False)
    parser_inference_validation.add_argument('--iteration', type=str)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_testing_data = subparsers.add_parser('inference_testing_data')
    parser_inference_testing_data.add_argument('--workspace', type=str)    
    parser_inference_testing_data.add_argument('--verified_only', action='store_true', default=False)
    parser_inference_testing_data.add_argument('--iteration', type=str)
    parser_inference_testing_data.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    
    args.filename = get_filename(__file__)
    
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)    
    logging = create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if args.mode == 'train':          
        train(args)
        
    elif args.mode == 'inference_validation':
        inference_validation(args)

    elif args.mode == 'inference_testing_data':
        inference_testing_data(args)
        
    else:
        raise Exception('Error!')