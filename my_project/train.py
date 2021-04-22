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
from test import init_test

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn
from random_word import RandomWords
import pandas as pd

from dataloaders import get_dataloaders, DatasetLoader, CamusResizedDataset, TTEDataset
#from Unet2D import Unet2D

import torch.optim as optim
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
import time
import os
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter
import torchgeometry
from logger_utils import *

np.random.seed(int(time.time()))

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
    return loss, acc, outputs
    

def check_accuracy(valid_dl, model, loss_fn, acc_fn, classes, tb_writer, seen_train_ex, other_logdir):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_dice = 0.0
    running_class_dice = np.zeros(classes)
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
                save_batch = save_batch_as_image(X_batch, Y_batch, outputs, seen_train_ex, "Validation", other_logdir)

    average_loss = running_loss / len(valid_dl.dataset)
    average_acc = running_acc / len(valid_dl.dataset)
    average_dice_sc = running_dice / len(valid_dl.dataset)
    average_dice_class_sc = running_class_dice / len(valid_dl.dataset)
    tb_writer.add_scalar("Val CE loss", average_loss, seen_train_ex)
    tb_writer.add_scalar("Val dice acc", average_dice_sc, seen_train_ex)
    tb_writer.add_scalar("Val px acc", average_acc, seen_train_ex)
    #tb_writer.add_custom_scalars("Val class dice acc", numpy_to_class_dict(average_dice_class_sc), seen_train_ex)
    for i,value in enumerate(average_dice_class_sc):
        tb_writer.add_scalar(f'Val dice class_{i+1}', value, seen_train_ex)
    print('{} Loss: {:.4f} PxAcc: {} Dice: {}'.format("Validation", average_loss, average_acc, average_dice_sc))
    return average_dice_sc, average_dice_class_sc



def numpy_to_class_dict(np_arr):
    ret_dict = {}
    for val in np_arr:
        ret_dict[f'Class {val+1}'] = val
    return ret_dict


def train(model, classes, train_dl, valid_dl, loss_fn, optimizer, scheduler, acc_fn, epochs, tb_writer, hparam_log, other_logdir):
    print(other_logdir)
    start = time.time()
    model.cuda()
    len_train_ds = len(train_dl.dataset)
    print("Len train ds")
    print(len_train_ds)
    seen_train_ex = 0

    avg_dice = 0.0
    avg_train_loss = 0.0
    best_acc = 0.0
    runs_without_improved_dice = 0
    highest_dice = 0.0
    seen_train_ex_highest_dice = 0
    hparam_log["hgst dice"] = 0.0
    hparam_log["hgst dice step"] = 0.0
    hparam_log["hgst dice tr CE loss"] = 0.0



    for epoch in range(epochs):
        save_batch = True
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
            loss, acc, outputs = train_step(X_batch, Y_batch, optimizer, model, loss_fn, acc_fn)
            running_acc  += acc*X_batch.size(0)
            running_loss += loss*X_batch.size(0)
            step += 1
            seen_train_ex += X_batch.size(0)
            tb_writer.add_scalar("Train CE loss", loss, seen_train_ex)
            tb_writer.add_scalar("Train px acc", acc, seen_train_ex)

            if save_batch:
                save_batch = save_batch_as_image(X_batch, Y_batch, outputs, seen_train_ex, "Train", other_logdir)
            if step % 25 == 0:
                print('Current step: {}  Loss: {}  Acc: {} '.format(step, loss, acc))

        avg_dice, avg_dice_cl = check_accuracy(valid_dl, model, loss_fn, acc_fn, classes, tb_writer, seen_train_ex, other_logdir)
        if avg_dice > highest_dice:
            highest_dice = avg_dice
            highest_dice_cl = avg_dice_cl

            hparam_log["hgst dice"] = highest_dice
            for i,dice in enumerate(avg_dice_cl):
                hparam_log[f'Class {i+1}'] = dice
            hparam_log["hgst dice step"] = seen_train_ex
            hparam_log["hgst dice tr CE loss"] = loss.item()
            runs_without_improved_dice = 0
            torch.save(model.state_dict(), os.path.join(other_logdir, "state_dict.pth"))

        else:
            runs_without_improved_dice +=1


        avg_train_loss = running_loss / len_train_ds
        avg_train_acc = running_acc / len_train_ds
        scheduler.step(avg_train_loss)
        print_epoch_stats(epoch, epochs, avg_train_loss, avg_train_acc)
        if runs_without_improved_dice > 20:
            print("Dice not improving for 12 epochs, abort training")
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
    lr_patience = cfg["lr_patience"]
    train_transforms = cfg["train_transforms"]
    val_transforms = cfg["val_transforms"]
    model_file = cfg["model"]
    dataset = cfg["dataset"]
    channel_ratio = cfg["channel_ratio"]
    cross_entr_weights = cfg["cross_entr_weights"]

    continue_training = False


    
    if "custom_logdir" in cfg:
        cust_logdir = cfg["custom_logdir"]
    else:
        cust_logdir = ""
    tb_logdir = os.path.join("logdir", "tensorboard", dataset, cust_logdir, model_file)
    other_logdir = os.path.join("logdir", "other", dataset, cust_logdir, model_file)
    print("other_logdir", other_logdir)

    try:
        try_number = len(os.listdir(tb_logdir))
    except:
        try_number = 0


    r = RandomWords()
    if continue_training:
        logdir_folder = "N1_None"
    else:
        random_word = r.get_random_word()

        logdir_folder = f'N{try_number}_{random_word}'
    tb_logdir = os.path.join(tb_logdir, logdir_folder)
    other_logdir = os.path.join(other_logdir, logdir_folder)
    os.makedirs(other_logdir, exist_ok=True)
    print("other_logdir:", other_logdir)
    print("tb_logdir:", tb_logdir)

    tb_writer = SummaryWriter(tb_logdir)


    train_loader, val_loader, classes = get_dataloaders(dataset, bs, train_transforms, val_transforms)

    model_path = os.path.join("models",dataset)
    model_import = import_model_from_path(model_file, model_path)


    unet = model_import.Unet2D(1,4, channel_ratio)
    if continue_training:
        unet.load_state_dict(torch.load(os.path.join(other_logdir, "state_dict.pth")))

    unet.cuda()

    loss_fn = torchgeometry.losses.dice_loss
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cross_entr_weights).cuda())
    #loss_fn2 = dice_loss
    #loss_fn3 = weighted_combined_loss(loss_fn, loss_fn2)
    opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True)


    train(unet, classes, train_loader, val_loader, loss_fn, opt, scheduler, mean_pixel_accuracy, epochs_val, tb_writer, hparam_log, other_logdir)

    test_dice_TTE, test_class_dice_TTE, test_dice_TEE, test_class_dice_TEE = init_test(cfg, logdir_folder, cust_logdir)
    hparam_log["test dice TTE"] = test_dice_TTE
    hparam_log["test dice TEE"] = test_dice_TEE
    for i,v in enumerate(test_class_dice_TTE):
        hparam_log[f'Test TTE Class{i+1}'] = v
    for i,v in enumerate(test_class_dice_TEE):
        hparam_log[f'Test TEE Class{i+1}'] = v

    
    all_stats_path = os.path.join("logdir", "all_train_stats.csv")
    df = pd.read_csv(all_stats_path)
    all_stats_dict = {"logdir": logdir_folder}
    all_stats_dict.update(cfg)
    all_stats_dict.update(hparam_log)
    write_out_transforms(all_stats_dict)
    df = df.append(all_stats_dict, ignore_index=True, sort=False)
    df = df.sort_values("hgst dice", ascending=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv(all_stats_path, index=False)

    del cfg["train_transforms"]
    del cfg["val_transforms"]
    del cfg["cross_entr_weights"]

    highest_dice = hparam_log["hgst dice"]

    dict_to_numpy(hparam_log)

    tb_writer.add_hparams(cfg, hparam_log)

    return highest_dice


if __name__ == "__main__":
    args = add_config_parser() 
    cfg = get_dict(args, print_config=True)
    init_train(cfg)
