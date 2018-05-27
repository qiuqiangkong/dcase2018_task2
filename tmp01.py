
import numpy as np
import time
import os
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

from utilities import create_folder, get_filename, create_logging, mat_2d_to_3d
from data_generator import *
import config
import average_precision


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()
    
    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)
    
    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        
        self.fc_final = nn.Linear(128, num_classes)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
    
        self.init_weights()
    
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        
    def forward(self, input):
        
        x = input.view((input.shape[0], 1, input.shape[1], input.shape[2]))
        
        drop_p = 0.2
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), p=drop_p, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), p=drop_p, training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), p=drop_p, training=self.training)
        x = F.dropout(F.relu(self.bn4(self.conv4(x))), p=drop_p, training=self.training)
        
        x = x.view((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        
        (x, _) = torch.max(x, dim=2)
        
        x = F.log_softmax(self.fc_final(x), dim=-1)
        
        return x
    
    
def move_x_to_gpu(x, cuda, volatile=False):
    x = torch.Tensor(x)
    if cuda:
        x = x.cuda()
    x = Variable(x, volatile=volatile)
    return x
    

def move_y_to_gpu(y, cuda, volatile=False):
    y = torch.LongTensor(y)
    if cuda:
        y = y.cuda()
    y = Variable(y, volatile=volatile)
    return y
    
    
def calculate_error(output, target):
    error = np.sum((output != target)) / float(target.shape[0])
    return error
    
    
def evaluate(model, generator, data_type, cuda):
    
    model.eval()
    
    time_steps = config.time_steps

    if data_type == 'train':
        gen_func = generator.generate_train_slices(max_audios=100)
    
    elif data_type == 'valid':
        gen_func = generator.generate_validate_slices(max_audios=None)
    
    losses = []
    predictions = []
    targets = []
    sorted_indices_list = []
    
    kmax = config.kmax
    
    for (slice_x, slice_y) in gen_func:
        
        xs = mat_2d_to_3d(slice_x, time_steps, time_steps // 2)
        xs = move_x_to_gpu(xs, cuda, volatile=True)

        ys = move_y_to_gpu(np.array([slice_y] * len(xs)), cuda, volatile=True)  
        
        output = model(xs)
        loss = F.nll_loss(output, ys)
        
        loss = loss.data.cpu().numpy()
        output = output.data.cpu().numpy()
        
        avg_output = np.mean(output, axis=0)
        sorted_indices = np.argsort(avg_output)[::-1][:kmax]
        prediction = sorted_indices[0]
        
        losses.append(loss)
        predictions.append(prediction)
        targets.append(slice_y)
        sorted_indices_list.append(sorted_indices)
    
    losses = np.array(losses)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
     
    # Error
    error = calculate_error(predictions, targets)
    acc = 1. - error
    loss = np.mean(losses)
    
    # mapk
    targets_list = [[e] for e in targets.tolist()]
    mapk = average_precision.mapk(targets_list, sorted_indices_list, k=kmax)
 
    return acc, mapk, loss
    
    
class ReduceLearningRate(object):
    def __init__(self, tolerance):
        self.queue = []
        self.max_queue_len = tolerance
        
    def check_reduce_learning_rate(self, loss):            
        
        if len(self.queue) < self.max_queue_len:
            self.queue.append(loss)
            return False
        else:
            del self.queue[0]
            self.queue.append(loss)
            
        if np.min(self.queue[1:]) < self.queue[0]:
            return False
        else:
            return True
        
    
    
def train(args):
    
    # Arguments & parameters
    workspace = args.workspace
    filename = args.filename
    cuda = True
    num_classes = config.num_classes
    
    if args.verified_only == 'True':
        verified_only = True
    elif args.verified_only == 'False':
        verified_only = False
        
    if args.validation == 'True':
        validation = True
    elif args.validation == 'False':
        validation = False
    
    batch_size = 128
    
    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'train.h5')
   
    model_dir = os.path.join(workspace, 'models', filename, 'verified_only={}, validation={}'.format(verified_only, validation))
    create_folder(model_dir)
    
    # Model
    model = CNN(num_classes)
    
    if cuda:
        model.cuda()
    
    # Data generator
    generator = Generator(hdf5_path=hdf5_path, batch_size=batch_size, time_steps=config.time_steps, verified_only=verified_only, validation=validation)
    # generator = Generator2(hdf5_path=hdf5_path, batch_size=batch_size, time_steps=config.time_steps, hop_frames=16, verified_only=verified_only, validation=validation)
    
    
    # Optimizer
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)

    reduce_lr = ReduceLearningRate(tolerance=5)
    
    logging.info("Train number: {}".format(len(generator.train_ids)))
    logging.info("Validate number: {}".format(len(generator.valid_ids)))

    iteration = 0
    
    bgn_train_time = time.time()
    
    for (batch_x, batch_y) in generator.generate():
        
        # Evaluate
        if iteration % 100 == 0:
            fin_train_time = time.time()
            eval_time = time.time()
            (tr_error, tr_mapk, tr_loss) = evaluate(model, generator, 'train', cuda)
            
            if validation:
                (va_error, va_mapk, va_loss) = evaluate(model, generator, 'valid', cuda)
        
            logging.info("Iteration: {}, train time: {:.3f}, eval time: {:.3f}"
                         .format(iteration, fin_train_time - bgn_train_time, time.time() - eval_time))
                         
            logging.info("Train acc: {:.3f}, Train mapk: {:.3f}, Train loss: {:.3f}"
                         .format(tr_error, tr_mapk, tr_loss))
            
            if validation:
                logging.info("Valid acc: {:.3f}, Valid mapk: {:.3f}, Valid loss: {:.3f}".format(va_error, va_mapk, va_loss))
                
            logging.info("------------")
            
            if reduce_lr.check_reduce_learning_rate(va_loss):
                learning_rate /= 10.
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
                logging.info("****** Learning rate reduced to {} ******".format(learning_rate))

            bgn_train_time = time.time()
        
        batch_x = move_x_to_gpu(batch_x, cuda)
        batch_y = move_y_to_gpu(batch_y, cuda)
        
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
            
            save_out_path = os.path.join(model_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            print("Save model to {}".format(save_out_path))
    
    
def inference(args):
    
    # Arguments & parameters
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    cuda = True
    
    num_classes = config.num_classes
    time_steps = config.time_steps
    ix_to_lb = config.ix_to_lb
    kmax = config.kmax
    
    # Paths
    model_path = os.path.join(workspace, 'models', filename, 'verified_only=False, validation=False', 'md_{}_iters.tar'.format(iteration))
    
    tr_hdf5_path = os.path.join(workspace, 'features', 'train.h5')
    te_hdf5_path = os.path.join(workspace, 'features', 'test.h5')
    
    submission_csv = os.path.join(workspace, 'submissions', filename, 'submission.csv')
    create_folder(os.path.dirname(submission_csv))
    
    # Model
    model = CNN(num_classes)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()
    
    # Data generator
    test_generator = TestGenerator(tr_hdf5_path, te_hdf5_path)
    
    # Write result to csv file
    f = open(submission_csv, 'w')
    
    f.write('fname,label\n')
    
    count = 0
    
    for (slice_x, audio_name) in test_generator.generate_test_slices():
        
        xs = mat_2d_to_3d(slice_x, time_steps, time_steps // 2)
        xs = move_x_to_gpu(xs, cuda, volatile=True)
        
        model.eval()
        output = model(xs)
        
        output = output.data.cpu().numpy()
        
        avg_output = np.mean(output, axis=0)
        sorted_indices = np.argsort(avg_output)[::-1][:kmax]
        prediction = sorted_indices[0]
        
        predicted_labels = [ix_to_lb[indice] for indice in sorted_indices]
        
        print(count, audio_name, predicted_labels)
        
        f.write('{},'.format(audio_name))
        for predicted_label in predicted_labels:
            f.write('{} '.format(predicted_label))
        f.write('\n')
        
        count += 1
        
        if count == 50:
            break
            
    f.write('{},{}\n'.format('0b0427e2.wav', 'Acoustic_guitar'))
    f.write('{},{}\n'.format('6ea0099f.wav', 'Acoustic_guitar'))
    f.write('{},{}\n'.format('b39975f5.wav', 'Acoustic_guitar'))
        
    f.close()
    
    print("Write to {}".format(submission_csv))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str)
    parser_train.add_argument('--verified_only', type=str, choices=['True', 'False'])
    parser_train.add_argument('--validation', type=str, choices=['True', 'False'])
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str)    
    parser_inference.add_argument('--iteration', type=str)
    
    args = parser.parse_args()
    print(args)
    
    filename = get_filename(__file__)
    logs_dir = os.path.join(args.workspace, 'logs', filename)
    create_folder(logs_dir)
    
    logging = create_logging(logs_dir, filemode='w')
    logging.info(os.path.abspath(__file__))
    logging.info(sys.argv)
    
    if args.mode == 'train':  
        args.filename = filename
        train(args)
        
    elif args.mode == 'inference':
        args.filename = filename
        inference(args)
        
    else:
        raise Exception('Error!')