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
from random_word import RandomWords
import pandas as pd

from dataloaders import *
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




def test(model, test_dl, loss_fn, acc_fn, classes, other_logdir, dataset):

    running_loss = 0.0
    running_acc = 0.0
    running_dice = 0.0
    running_class_dice = np.zeros(classes)
    
    
    with torch.no_grad():
        for i,data in enumerate(test_dl):
            X_batch, Y_batch = data
            X_batch = X_batch.cuda()
            Y_batch = Y_batch.cuda()
            cur_batch_sz = X_batch.size(0)
            outputs = model(X_batch)
            loss = loss_fn(outputs, Y_batch.long())
            acc = acc_fn(outputs, Y_batch)
            dice_score, dice_class_scores = mean_dice_score(outputs, Y_batch, classes)
            running_acc += acc * cur_batch_sz
            running_loss += loss * cur_batch_sz
            running_dice += dice_score * cur_batch_sz
            running_class_dice += dice_class_scores * cur_batch_sz
            save_batch_as_image(X_batch, Y_batch, outputs, i, "Test" + "" + dataset, other_logdir)

    len_dl = len(test_dl.dataset)
    average_loss = running_loss / len_dl
    average_acc = running_acc / len_dl
    average_dice_sc = running_dice / len_dl
    average_dice_class_sc = running_class_dice / len_dl
    print('{} Loss: {:.4f} PxAcc: {} Dice: {}'.format("Validation", average_loss, average_acc, average_dice_sc))
    return average_dice_sc, average_dice_class_sc





def init_test(cfg, model_folder, hparam_search_dir=""):
    print("Model folder")
    print(model_folder)

    bs = 4
    val_transforms = cfg["val_transforms"]
    model_file = cfg["model"]
    dataset = cfg["dataset"]
    channel_ratio = cfg["channel_ratio"]
    cross_entr_weights = cfg["cross_entr_weights"]

    other_logdir = os.path.join("logdir", "other", dataset, hparam_search_dir, model_file)
    other_logdir = os.path.join(other_logdir, model_folder)
    model_state_dict_path = os.path.join(other_logdir, "state_dict.pth")
    print("Save model path")
    print(model_state_dict_path)

    model_path = os.path.join("models",dataset)
    model_import = import_model_from_path(model_file, model_path)


    unet = model_import.Unet2D(1,4, channel_ratio)
    unet.load_state_dict(torch.load(model_state_dict_path))
    unet.cuda()

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cross_entr_weights).cuda())

    test_loader_TTE, classes = get_test_loader(dataset, bs, val_transforms)
    test_loader_TEE,_ = get_test_loader("TEE", bs, val_transforms)


    average_dice_sc, average_dice_class_sc = test(unet, test_loader_TTE, loss_fn, mean_pixel_accuracy, classes, other_logdir, "TTE")
    average_dice_sc_TEE, average_dice_class_sc_TEE = test(unet, test_loader_TEE, loss_fn, mean_pixel_accuracy, classes-1, other_logdir, "TEE")

    return average_dice_sc, average_dice_class_sc, average_dice_sc_TEE, average_dice_class_sc_TEE















if __name__ == '__main__':
    args = add_config_parser_with_model_folder()
    cfg = get_dict(args, print_config=True)
    model_folder = args.model_folder
    init_test(cfg, model_folder)

