# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import csv

sys.path.append('../../')
from utils.dataset import read_data
from utils.transformer import Transformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import time
from libs.print_para import print_para

torch.manual_seed(22)

parse = argparse.ArgumentParser()
parse.add_argument('-cuda', type=int, default=1)
# parse.add_argument('-notes', type=str, default='221021')
parse.add_argument('-batch_size', type=int, default=32, help='batch size')
parse.add_argument('-epoch_size', type=int, default=300, help='epochs')
parse.add_argument('-d_model', type=int, default=18)
parse.add_argument('-dk_s', type=int, default=20)
parse.add_argument('-dk_t', type=int, default=20)
parse.add_argument('-nheads_s', type=int, default=6)
parse.add_argument('-nheads_t', type=int, default=4)
parse.add_argument('-d_inner', type=int, default=128)
parse.add_argument('-layers', type=int, default=3)
parse.add_argument('-close_size', type=int, default=3)  # *******

parse.add_argument('-nb_flow', type=int, default=1)

parse.add_argument('-height', type=int, default=32)
parse.add_argument('-width', type=int, default=32)

parse.add_argument('-meta', type=int, default=0)
parse.add_argument('-cross', type=int, default=0)
parse.add_argument('-cluster', type=int, default=3)  # default-3

parse.add_argument('-loss', type=str, default='l1', help='l1 | l2')  # default-l1
parse.add_argument('-lr', type=float, default=1e-3)  # default-0.001
parse.add_argument('-weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

parse.add_argument('-test_size', type=int, default=120)

parse.add_argument('-fusion', type=int, default=1)

parse.set_defaults(crop=True)
parse.add_argument('-train', dest='train', action='store_true')

parse.set_defaults(train=True)
parse.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
parse.add_argument('-adam', dest='adam', help='use adam. Not recommended', action='store_true')

opt = parse.parse_args()
device = torch.device("cuda:{}".format(opt.cuda))

if opt.loss == 'l1':
    criterion = nn.L1Loss().cuda()
elif opt.loss == 'l2':
    criterion = nn.MSELoss().cuda()

model_ver = 'm_ver1'
path_name = 'bj_taxi_result/' + model_ver

if not os.path.exists(path_name):
    os.makedirs(path_name)
else:
    print('path already exists.')

file_name = 'bj_taxi'
hdf5_file = os.path.join(path_name, "%s.h5" % file_name)


def get_optim(lr):
    if opt.adam:
        optimizer = optim.Adam(model.parameters(), weight_decay=opt.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(model.parameters(), weight_decay=opt.l2, lr=lr, momentum=0.9)

    # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.05,
    #                             verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * opt.epoch_size, 0.75 * opt.epoch_size],
                                                     gamma=0.1)
    return optimizer, scheduler


def predict(test_type='train'):
    predictions = []
    ground_truth = []
    attn_s_list, attn_t_list = [], []
    loss = []
    model.eval()
    # print(opt.model_filename + '.pt')
    model.load_state_dict(torch.load(opt.model_filename + '.pt'))

    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader

    with torch.no_grad():
        for idx, (c, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = c.float().to(device)
            # print('x.shape:',x.shape)
            # print('x[0]:',x[0])
            y = target.float().to(device)
            pred, weight = model(x)
            pred = pred[:, 0, :, :]
            predictions.append(pred.data.cpu())
            ground_truth.append(target.data)
            loss.append(criterion(pred, y).item())

    final_predict = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    print(
        "Shape of final prediction is {}, shape of ground truth is {}".format(final_predict.shape, ground_truth.shape))

    ground_truth = mmn.inverse_transform(ground_truth)
    final_predict = mmn.inverse_transform(final_predict)
    return final_predict, ground_truth, weight


def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length = len(dataloader)
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


if __name__ == '__main__':
    path = '../../data/BJ16_In.h5'

    X, y, mmn = read_data(path, opt)
    print("X.shape", X.shape)
    print("y.shape", y.shape)

    samples, sequences, channels, height, width = X.shape

    x_train, x_test = X[:-opt.test_size], X[-opt.test_size:]

    y_tr = y[:-opt.test_size]
    y_te = y[-opt.test_size:]
    # print("x_tr:", x_train.shape)
    # print("x_te:", x_test.shape)
    print("y_tr:", y_tr.shape)
    print("y_te:", y_te.shape)

    prediction_ct = 0
    truth_ct = 0
    opt.model_filename = path_name + '/' + model_ver

    y_train = y_tr
    y_test = y_te

    if (opt.meta == 1) & (opt.cross == 1):
        train_data = list(zip(*[x_train, meta_train, cross_train, y_train]))
        test_data = list(zip(*[x_test, meta_test, cross_test, y_test]))
    elif (opt.meta == 1) & (opt.cross == 0):
        train_data = list(zip(*[x_train, meta_train, y_train]))
        test_data = list(zip(*[x_test, meta_test, y_test]))
    elif (opt.cross == 1) & (opt.meta == 0):
        train_data = list(zip(*[x_train, cross_train, y_train]))
        test_data = list(zip(*[x_test, cross_test, y_test]))
    elif (opt.meta == 0) & (opt.cross == 0):
        train_data = list(zip(*[x_train, y_train]))
        test_data = list(zip(*[x_test, y_test]))

    # split the training data into train and validation
    train_idx, valid_idx = train_valid_split(train_data, 0.1)
    # print(train_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # print(train_sampler)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler,
                              num_workers=0, pin_memory=True)
    valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler,
                              num_workers=0, pin_memory=True)
    # print(test_data.index)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    # print(train_loader)
    input_shape = X.shape
    meta_shape = []
    cross_shape = []

    model = Transformer(input_shape,
                        meta_shape,
                        cross_shape,
                        nb_flows=opt.nb_flow,
                        fusion=opt.fusion,
                        maps=(opt.meta + opt.cross + 1),
                        d_model=opt.d_model,
                        dk_t=opt.dk_t,
                        dk_s=opt.dk_s,
                        nheads_spatial=opt.nheads_s,
                        nheads_temporal=opt.nheads_t,
                        d_inner=opt.d_inner,
                        layers=opt.layers,
                        flags_meta=opt.meta,
                        flags_cross=opt.cross
                        ).to(device)

    model.load_state_dict(torch.load(opt.model_filename + '.pt'))

    optimizer = optim.Adam(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * opt.epoch_size,
                                                                 0.75 * opt.epoch_size, 0.9 * opt.epoch_size],
                                                     gamma=0.1)

    pred, truth, weight = predict('test')

    prediction_ct += pred[:, :, :].reshape(opt.test_size, -1)
    truth_ct += truth[:, :, :].reshape(opt.test_size, -1)
    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset('pred', data=prediction_ct)
        f.create_dataset('truth', data=truth_ct)

    Y = truth_ct.ravel()
    Y_hat = prediction_ct.ravel()
    print('Final R^2 Score: {:.4f}'.format(metrics.r2_score(Y, Y_hat)))
    print('Final Variance Score: {:.4f}'.format(metrics.explained_variance_score(Y, Y_hat)))

    if not os.path.exists('bj_taxi_result/' + model_ver + '/result.txt'):
        with open('bj_taxi_result/' + model_ver + '/result.txt', 'a') as f:
            f.write('model ' + model_ver)
            f.write('\n')
            f.write(print_para(model))
            f.write('\n')
            f.write('RMSE:{:0.4f}'.format(metrics.mean_squared_error(prediction_ct.ravel(), truth_ct.ravel()) ** 0.5))
            f.write('\n')
            f.write('Final MAE:{:0.4f}'.format(metrics.mean_absolute_error(prediction_ct.ravel(), truth_ct.ravel())))
            f.write('\n')
            f.write('R^2 Score: {:.4f}'.format(metrics.r2_score(Y, Y_hat)))
            f.write('\n')
            f.write('Variance Score: {:.4f}'.format(metrics.explained_variance_score(Y, Y_hat)))
            f.write('\n')
            f.close()
