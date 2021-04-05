import torch
import numpy as np


def dice_score(segmentation, ground_truth):
    return np.sum(seg[ground_truth==1])*2.0 / (np.sum(segmentation) + np.sum(ground_truth))

def image_stats(img):
    data_type = type(img[0][0])
    img_width = np.shape(img)[0]
    img_height = np.shape(img)[1]
    max_pix = np.max(img)
    min_pix = np.min(img)
    img_mean = np.mean(img)
    img_std = np.std(img)
    print(f'Type: {data_type}, Width: {img_width}, Height: {img_height}, Max: {max_pix}, Min: {min_pix}, Mean: {img_mean}, Std: {img_std}')

def tensor_stats(tensor_in):
    tensor = tensor_in.clone()
    tensor = tensor.double()
    shape = tensor.shape
    tensor_max = torch.max(tensor)
    tensor_min = torch.min(tensor)
    tensor_mean = torch.mean(tensor)
    tensor_std = torch.std(tensor)
    print(f"Tensor stats: Shape: {shape} Max: {tensor_max}, Min: {tensor_min}, Mean: {tensor_mean}, Std: {tensor_std}")


def get_mask_from_tensor(tensor, index, mask_index):
    tensor_cp = tensor.clone().cpu()
    tensor_masks = tensor_cp[index]
    tensor_mask = tensor_masks[mask_index]
    np_mask = tensor_mask.numpy()
    print("np mask in get mask")
    image_stats(np_mask)
    return np_mask


