import torch
import numpy as np
import matplotlib.pyplot as plt


def convert_tensor_to_RGB(network_output):
    x = torch.FloatTensor([[.0, .0, .0], [1.0, .0, .0], [.0, .0, 1.0], [.0, 1.0, .0]])
    converted_tensor = torch.nn.functional.embedding(network_output, x).permute(2,0,1)
    return converted_tensor


def dice_scores(segmentation, ground_truth, classes):
    dice_scores = []
    for i in range(1,classes+1):
        binary_gt = (ground_truth == i).astype(np.uint8)
        binary_seg = (segmentation == i).astype(np.uint8)
        intersect = np.logical_and(binary_gt, binary_seg)
        sum_binary_gt = np.sum(binary_gt)
        sum_binary_seg = np.sum(binary_seg)
        if sum_binary_gt == 0:
            continue
        class_dice_score = np.sum(intersect)*2 / (sum_binary_gt+sum_binary_seg)
        dice_scores.append(class_dice_score)
    dice_scores = np.array(dice_scores)
    return dice_scores







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


def dice_loss(logits, target):
    input = torch.functional.F.softmax(logits, 1)
    smooth = 1.
    
    input = input[:,1,:,:]
    #print(input.shape)
    #print(target.shape)


    iflat = torch.reshape(input, (-1,))
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def weighted_combined_loss(loss_fn1, loss_fn2, weight=0.5):
    def combined_loss(pred, Y):
        return weight*loss_fn1(pred,Y) + (1-weight)*loss_fn2(pred,Y)
    return combined_loss



def mean_dice_score(pred_batch, Y_batch, classes):
    assert(pred_batch.size(0) == Y_batch.size(0))
    cumulative_scores = np.zeros(classes)
    for b_idx in range(pred_batch.size(0)):
        mask = predb_to_mask(pred_batch, b_idx).numpy()
        gt_tensor = Y_batch[b_idx].clone()
        gt = gt_tensor.cpu().numpy()

        batch_dice_score = dice_scores(mask, gt, classes)

        cumulative_scores += batch_dice_score

    avg_dice_scores = cumulative_scores / pred_batch.size(0)
    avg_dice_score = np.average(avg_dice_scores)
    return avg_dice_score, avg_dice_scores

def mean_pixel_accuracy(pred_batch, Y_batch):
    return (pred_batch.argmax(dim=1) == Y_batch.cuda()).float().mean()

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(pred_batch, idx):
    p = torch.functional.F.softmax(pred_batch[idx], 0)
    return p.argmax(0).cpu()


