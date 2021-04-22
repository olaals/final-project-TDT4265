import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt


def get_np_image(img_path):
    pil_img = Image.open(img_path).convert('L')
    img_np = np.array(pil_img)
    return img_np

def load_patient_input_gt(TEE_path, patient):
    input_img_path = os.path.join(TEE_path, "train_gray", f'gray_{patient}.jpg')
    gt_img_path = os.path.join(TEE_path, "train_gt", f'gt_gt_{patient}.tif')
    input_img = get_np_image(input_img_path)
    gt_img = get_np_image(gt_img_path)
    gt_img[gt_img==255] = 2.0
    gt_img[gt_img==127] = 1.0

    gt_img = gt_img[~np.all(input_img == 0, axis=1)]
    input_img = input_img[~np.all(input_img == 0, axis=1)]
    gt_img = np.rot90(gt_img,1)
    input_img = np.rot90(input_img,1)
    gt_img = gt_img[~np.all(input_img == 0, axis=1)]
    input_img = input_img[~np.all(input_img == 0, axis=1)]

    return input_img, gt_img

def get_all_TEE_images_and_gt(TEE_path):

    train_img_files = os.listdir(os.path.join(TEE_path, 'train_gray'))
    input_gt_tuples = []
    for train_img_file in train_img_files:
        remove_ext = os.path.splitext(train_img_file)[0]
        split_string = remove_ext.split("_")
        patient = f'{split_string[1]}_{split_string[2]}_{split_string[3]}'
        input_gt_tuple = load_patient_input_gt(TEE_path, patient)
        input_gt_tuples.append(input_gt_tuple)
    return input_gt_tuples

















if __name__ == '__main__':
    TEE_path = "/home/ola/projects/final-project-TDT4265/my_project/datasets/TEE"
    TEE_input_img = "/home/ola/projects/final-project-TDT4265/my_project/datasets/TEE/train_gray/gray_D102722_J4AADLHQ_129.jpg"
    TEE_gt_img = "/home/ola/projects/final-project-TDT4265/my_project/datasets/TEE/train_gt/gt_gt_D102722_J4AADLHQ_129.tif" 

    input_gt_list = get_all_TEE_images_and_gt(TEE_path)
    print(len(input_gt_list))

    for input_img, gt_img in input_gt_list:
        print(input_img.shape)
        print(gt_img.shape)
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(input_img)
        ax[1].imshow(gt_img)
        plt.show()








