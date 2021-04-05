import sys
sys.path.append("utils")
sys.path.append("models")
from file_io import get_dict, add_config_parser
from train_utils import *

import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

from DatasetLoader import DatasetLoader, CamusResizedDataset
from Unet2D import Unet2D

import torch.optim as optim
import torchvision
from tqdm import tqdm
import time
import os
from importlib import import_module

PRINT_DEBUG = False

def train_step(X_batch, Y_batch, optimizer, model, loss_fn, acc_fn):
    X_batch = X_batch.cuda()
    Y_batch = Y_batch.cuda()
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = loss_fn(outputs, Y_batch)
    loss.backward()
    optimizer.step()
    acc = acc_fn(outputs, Y_batch)
    return loss, acc
    

def check_accuracy(valid_dl, model, loss_fn, acc_fn):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in valid_dl:
            X_batch = X_batch.cuda()
            Y_batch = Y_batch.cuda()
            cur_batch_sz = X_batch.size(0)
            outputs = model(X_batch)
            loss = loss_fn(outputs, Y_batch.long())
            acc = acc_fn(outputs, Y_batch)
            running_acc  += acc * cur_batch_sz
            running_loss += loss * cur_batch_sz
    average_loss = running_loss / len(valid_dl.dataset)
    average_acc = running_acc / len(valid_dl.dataset)
    print('{} Loss: {:.4f} Acc: {}'.format("Validation", average_loss, average_acc))
    return average_loss, average_acc


def print_epoch_stats(epoch, epochs, avg_train_loss, avg_train_acc):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)
    print('{} Loss: {:.4f} Acc: {}'.format("Train", avg_train_loss, avg_train_acc))
    print('-' * 10)


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()
    len_train_ds = len(train_dl.dataset)

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_acc = 0.0
        step = 0
        # iterate over data
        for X_batch, Y_batch in train_dl:
            loss, acc = train_step(X_batch, Y_batch, optimizer, model, loss_fn, acc_fn)
            running_acc  += acc*X_batch.size(0)
            running_loss += loss*X_batch.size(0)
            step += 1
            if step % 5 == 0:
                print('Current step: {}  Loss: {}  Acc: {} '.format(step, loss, acc))

        avg_val_loss, avg_val_acc = check_accuracy(valid_dl, model, loss_fn, acc_fn)
        

        avg_train_loss = running_loss / len_train_ds
        avg_train_acc = running_acc / len_train_ds
        train_loss.append(avg_train_loss)
        valid_loss.append(avg_val_acc)
        print_epoch_stats(epoch, epochs, avg_train_loss, avg_train_acc)


    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    

def acc_metric(predb, yb):
    #print(type(predb))
    #print(predb.shape)
    #print(torch.max(predb))
    #print(torch.min(predb)) 
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def main ():
    #enable if you want to see some plotting
    visual_debug = True

    args = add_config_parser() 
    cfg = get_dict(args, print_config=True)

    #batch size
    bs = cfg["batch_size"]

    #epochs
    epochs_val = cfg["epochs"]

    #learning rate
    learn_rate = cfg["learning_rate"]

    train_dir = cfg["train_dir"]
    val_dir = cfg["val_dir"]
    test_dir = cfg["test_dir"]

    train_transforms = cfg["train_transforms"]
    val_transforms = cfg["val_transforms"]

    #sets the matplotlib display backend (most likely not needed)
    mp.use('TkAgg', force=True)

    train_ds = CamusResizedDataset(train_dir, transforms=train_transforms)
    val_ds = CamusResizedDataset(val_dir, transforms=val_transforms)
    test_ds = CamusResizedDataset(test_dir)


    #split the training dataset and initialize the data loaders
    train_data = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    valid_data = DataLoader(dataset=val_ds, batch_size=bs, shuffle=True)



    if visual_debug:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(train_ds.get_np_img(150))
        ax[1].imshow(train_ds.get_np_mask(150))
        plt.show()

    #xb, yb = next(iter(train_data))
    #print (xb.shape, yb.shape)

    # build the Unet2D with one channel as input and 2 channels as output
    unet = Unet2D(1,2)
    unet.cuda()

    #loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)

    #do some training
    train_loss, valid_loss = train(unet, train_data, valid_data, loss_fn, opt, acc_metric, epochs=epochs_val)

    #plot training and validation losses
    if visual_debug:
        print("Visual debug")
        plt.figure(figsize=(10,8))
        plt.plot(train_loss, label='Train loss')
        plt.plot(valid_loss, label='Valid loss')
        plt.legend()
        plt.show()

    #predict on the next train batch (is this fair?)
    xb, yb = next(iter(train_data))
    with torch.no_grad():
        predb = unet(xb.cuda())

    #show the predicted segmentations
    max_images_to_show = 4
    if visual_debug:
        images_to_show = min(max_images_to_show, bs)
        fig, ax = plt.subplots(images_to_show,5, figsize=(15,bs*5))
        for i in range(images_to_show):
            train_img = batch_to_img(xb,i)
            print("train_img")
            image_stats(train_img)
            gt_img = yb[i].numpy()
            print("gt_img")
            image_stats(gt_img)

            bg_mask = get_mask_from_tensor(predb, i,0)
            pred_mask = get_mask_from_tensor(predb, i,1)

            pred_img = predb_to_mask(predb, i).numpy()
            print("pred_img")
            image_stats(pred_img)

            ax[i,0].imshow(train_img)
            ax[i,1].imshow(gt_img)
            ax[i,2].imshow(pred_img)
            ax[i,3].imshow(bg_mask)
            ax[i,4].imshow(pred_mask)

        plt.show()

if __name__ == "__main__":
    main()
