import os
import torchvision
import torch
from train_utils import *


def write_out_transforms(config):
    for transform in config["train_transforms"]:
        config[str(transform).split('(')[0]] = transform
    for transform in config["val_transforms"]:
        config[str(transform).split('(')[0]] = transform

def save_batch_as_image(X_batch, Y_batch, outputs, seen_train_ex, label, other_logdir, num_rows_to_plot=3):
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
    #grid = torchvision.transforms.functional.resize(grid, 256)
    save_dir = os.path.join(other_logdir, label)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{format(seen_train_ex, "09d")}.png')

    torchvision.utils.save_image(grid,save_path)
    return save_batch


def save_batch_tensorboard(X_batch, Y_batch, outputs, seen_train_ex, tb_writer, label, num_rows_to_plot=3):
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
    #grid = torchvision.transforms.functional.resize(grid, 256)
    tb_writer.add_image(label, grid, global_step=seen_train_ex)
    return save_batch


def print_epoch_stats(epoch, epochs, avg_train_loss, avg_train_acc):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)
    print('{} Loss: {:.4f} PxAcc: {}'.format("Train", avg_train_loss, avg_train_acc))
    print('-' * 10)

