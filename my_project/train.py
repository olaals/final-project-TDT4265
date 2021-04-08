import sys
sys.path.append("utils")
sys.path.append("models")
from file_io import *
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

from dataloaders import get_dataloaders, DatasetLoader, CamusResizedDataset, TTEDataset
#from Unet2D import Unet2D

import torch.optim as optim
import torchvision
from tqdm import tqdm
import time
import os
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter
import torchgeometry

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
    

def convert_tensor_to_RGB(network_output):
    x = torch.FloatTensor([[.0, .0, .0], [1.0, .0, .0], [.0, .0, 1.0], [.0, 1.0, .0]])
    converted_tensor = torch.nn.functional.embedding(network_output, x).permute(2,0,1)
    return converted_tensor


def check_accuracy(valid_dl, model, loss_fn, acc_fn, classes, tb_writer, seen_train_ex):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_dice = 0.0
    running_class_dice = np.zeros(classes)
    num_rows_to_plot = 5
    save_batch = True
    with torch.no_grad():
        for X_batch, Y_batch in valid_dl:
            X_batch = X_batch.cuda()
            Y_batch = Y_batch.cuda()
            cur_batch_sz = X_batch.size(0)
            outputs = model(X_batch)
            loss = loss_fn(outputs, Y_batch.long())
            acc = acc_fn(outputs, Y_batch)
            dice_score, dice_class_scores = mean_dice_score(outputs, Y_batch, classes)
            running_acc  += acc * cur_batch_sz
            running_loss += loss * cur_batch_sz
            running_dice += dice_score * cur_batch_sz
            running_class_dice += dice_class_scores * cur_batch_sz
            if save_batch:
                save_batch = False
                np_grid = []
                num_rows_to_plot = min(X_batch.size(0), num_rows_to_plot)
                for i in range(num_rows_to_plot):
                    input_img = X_batch[i].cpu().float()
                    input_img = torch.cat([input_img, input_img, input_img])
                    mask = predb_to_mask(outputs.clone(), i)
                    mask = convert_tensor_to_RGB(mask)
                    gt = Y_batch[i].cpu()
                    gt = convert_tensor_to_RGB(gt)
                    np_grid.append(input_img)
                    np_grid.append(mask)
                    np_grid.append(gt)

                grid = torchvision.utils.make_grid(np_grid, nrow=num_rows_to_plot)
                tb_writer.add_image("Validation: input, pred, gt", grid, global_step=seen_train_ex)

    average_loss = running_loss / len(valid_dl.dataset)
    average_acc = running_acc / len(valid_dl.dataset)
    average_dice_sc = running_dice / len(valid_dl.dataset)
    average_dice_class_sc = running_class_dice / len(valid_dl.dataset)
    tb_writer.add_scalar("Val CE loss", average_loss, seen_train_ex)
    tb_writer.add_scalar("Val dice acc", average_dice_sc, seen_train_ex)
    tb_writer.add_scalar("Val px acc", average_acc, seen_train_ex)
    tb_writer.add_scalars("Val class dice acc", numpy_to_class_dict(average_dice_class_sc), seen_train_ex)
    print('{} Loss: {:.4f} PxAcc: {} Dice: {}'.format("Validation", average_loss, average_acc, average_dice_sc))
    return average_dice_sc 


def print_epoch_stats(epoch, epochs, avg_train_loss, avg_train_acc):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)
    print('{} Loss: {:.4f} PxAcc: {}'.format("Train", avg_train_loss, avg_train_acc))
    print('-' * 10)

def numpy_to_class_dict(np_arr):
    ret_dict = {}
    for val in np_arr:
        ret_dict[f'Class {val+1}'] = val
    return ret_dict


def train(model, classes, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1, tb_writer=None, hparam_log=None):
    start = time.time()
    model.cuda()
    len_train_ds = len(train_dl.dataset)
    print("Len train ds")
    print(len_train_ds)
    seen_train_ex = 0

    best_acc = 0.0
    runs_without_improved_dice = 0
    highest_dice = 0.0
    seen_train_ex_highest_dice = 0
    hparam_log["hgst dice"] = 0.0
    hparam_log["hgst dice step"] = 0.0
    hparam_log["hgst dice tr CE loss"] = 0.0

    for epoch in range(epochs):
        model.train()
        weight = epoch/epochs
        print("weight", weight)
        #loss_fn = weighted_combined_loss(nn.CrossEntropyLoss(), dice_loss, weight)
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
            seen_train_ex += X_batch.size(0)
            tb_writer.add_scalar("Train CE loss", loss, seen_train_ex)
            tb_writer.add_scalar("Train px acc", acc, seen_train_ex)
            if step % 5 == 0:
                print('Current step: {}  Loss: {}  Acc: {} '.format(step, loss, acc))

        avg_dice = check_accuracy(valid_dl, model, loss_fn, acc_fn, classes, tb_writer, seen_train_ex)
        if avg_dice > highest_dice:
            highest_dice = avg_dice
            hparam_log["hgst dice"] = highest_dice
            hparam_log["hgst dice step"] = seen_train_ex
            hparam_log["hgst dice tr CE loss"] = loss
        else:
            runs_without_improved_dice +=1


        avg_train_loss = running_loss / len_train_ds
        avg_train_acc = running_acc / len_train_ds
        print_epoch_stats(epoch, epochs, avg_train_loss, avg_train_acc)
        if runs_without_improved_dice > 5:
            print("Dice not improving for 5 epochs, abort training")
            break

    hparam_log["last step"] = seen_train_ex
    hparam_log["last dice"] = avg_dice
    hparam_log["last train loss"] = avg_train_loss
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    




def dict_to_numpy(hparam_dict):
    hparam_dict["last train loss"] = hparam_dict["last train loss"].item()
    for key in hparam_dict:
        try:
            hparam_dict[key] = hparam_dict[key].item()
        except:
            pass
        try:
            hparam_dict[key] = hparam_dict[key].detach().cpu().numpy()
        except:
            pass


def init_train(cfg):

    hparam_log = {}

    bs = cfg["batch_size"]
    epochs_val = cfg["epochs"]
    learn_rate = cfg["learning_rate"]
    train_transforms = cfg["train_transforms"]
    val_transforms = cfg["val_transforms"]
    model_file = cfg["model"]
    dataset = cfg["dataset"]
    channel_ratio = cfg["channel_ratio"]
    cross_entr_weights = cfg["cross_entr_weights"]

    h_params = {"bs": bs, "lr": learn_rate}


    if "custom_logdir" in cfg:
        cust_logdir = cfg["custom_logdir"]
    else:
        cust_logdir = ""
    logdir = os.path.join("tensorboard", dataset, cust_logdir, model_file)

    try:
        try_number = len(os.listdir(logdir))
    except:
        try_number = 0

    logdir_folder = f'N{try_number}_bs{bs}_lr{learn_rate}'
    logdir = os.path.join(logdir, logdir_folder)

    tb_writer = SummaryWriter(logdir)

    train_loader, val_loader, classes = get_dataloaders(dataset, bs, train_transforms, val_transforms)

    model_path = os.path.join("models",dataset)
    model_import = import_model_from_path(model_file, model_path)

    unet = model_import.Unet2D(1,4, channel_ratio)
    unet.cuda()

    loss_fn = torchgeometry.losses.dice_loss
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cross_entr_weights).cuda())
    #loss_fn2 = dice_loss
    #loss_fn3 = weighted_combined_loss(loss_fn, loss_fn2)
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)

    train(unet, classes, train_loader, val_loader, loss_fn, opt, mean_pixel_accuracy, epochs=epochs_val, tb_writer=tb_writer, hparam_log=hparam_log)


    dict_to_numpy(hparam_log)
    highest_dice = hparam_log["hgst dice"]
    del hparam_log["hgst dice"]
    highest_dice_train_loss = hparam_log["hgst dice tr CE loss"]
    del hparam_log["hgst dice tr CE loss"]
    h_params.update(hparam_log)
    print(h_params)

    tb_writer.add_hparams(h_params, {"highest dice": highest_dice, "hgst dice tr loss":highest_dice_train_loss})

    return highest_dice


if __name__ == "__main__":
    args = add_config_parser() 
    cfg = get_dict(args, print_config=True)
    init_train(cfg)
