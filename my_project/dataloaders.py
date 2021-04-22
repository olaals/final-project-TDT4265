import os
import sys
sys.path.append("utils")
from camus_resized_loader import load_input_gt_dir
from mhd_loader import get_all_patients_imgs
import numpy as np
import torch
import albumentations as A

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import matplotlib.pyplot as plt
from TEE_loader import get_all_TEE_images_and_gt

def get_dataloaders(dataset, batch_size, train_transforms, val_transforms):
    train_path = os.path.join("datasets", dataset, "train")
    val_path = os.path.join("datasets", dataset, "val")
    if dataset == "TTE":
        train_ds = TTEDataset(train_path, transforms=train_transforms)
        val_ds = TTEDataset(val_path, transforms=val_transforms)
        train_loader = DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size, shuffle=False)
        classes = 3

    if dataset == "CAMUS_resized":
        train_ds = CamusResizedDataset(train_path, transforms=train_transforms)
        val_ds = CamusResizedDataset(train_path, transforms=val_transforms)
        train_loader = DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size, shuffle=False)
        classes = 1

    return train_loader, val_loader, classes

def get_test_loader(dataset, batch_size, val_transforms):
    if dataset == "TTE":
        test_path = os.path.join("datasets", dataset, "test")
        test_ds = TTEDataset(test_path, transforms=val_transforms)
        test_loader = DataLoader(test_ds, batch_size, shuffle=False)
        classes = 3
    elif dataset == 'TEE':
        test_path = os.path.join("datasets", dataset)
        test_ds = TEEDataset(test_path, val_transforms)
        test_loader = DataLoader(test_ds, batch_size, shuffle=False)
        classes = 3

    return test_loader, classes




class TEEDataset(Dataset):
    def __init__(self, TEE_dir, 
            transforms
            ):
        super().__init__()
        self.transforms = transforms
        self.input_gt_imgs_list = get_all_TEE_images_and_gt(TEE_dir)
        print(len(self.input_gt_imgs_list))


    def __len__(self):
        return len(self.input_gt_imgs_list)

    def __getitem__(self, index):
        input_img, gt_img = self.input_gt_imgs_list[index]


        if self.transforms:
            transformed = self.transforms(image=input_img,mask=gt_img)
            input_img = transformed["image"]
            gt_img = transformed["mask"]

        input_img = np.expand_dims(input_img,0)
        input_img = torch.tensor(input_img, dtype=torch.float32)
        gt_img = torch.tensor(gt_img, dtype=torch.torch.int64)
        
        return input_img, gt_img

    def get_np_img(self, index):
        return self.input_gt_imgs_list[index][0]

    def get_np_mask(self, index):
        return self.input_gt_imgs_list[index][1]


class TTEDataset(Dataset):
    def __init__(self, input_gt_dir, isotropic_imgs=False, 
            transforms = A.Compose([
                #A.Resize(500,500),
                A.HorizontalFlip(p=.5),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.3, rotate_limit=30, p=0.9),
                ])
            ):
        super().__init__()
        self.transforms = transforms
        self.input_gt_imgs_list = get_all_patients_imgs(input_gt_dir, isotropic_imgs)
        print(len(self.input_gt_imgs_list))


    def __len__(self):
        return len(self.input_gt_imgs_list)

    def __getitem__(self, index):
        input_img, gt_img = self.input_gt_imgs_list[index]


        if self.transforms:
            transformed = self.transforms(image=input_img,mask=gt_img)
            input_img = transformed["image"]
            gt_img = transformed["mask"]

        input_img = np.expand_dims(input_img,0)
        input_img = torch.tensor(input_img, dtype=torch.float32)
        gt_img = torch.tensor(gt_img, dtype=torch.torch.int64)
        
        return input_img, gt_img

    def get_np_img(self, index):
        return self.input_gt_imgs_list[index][0]

    def get_np_mask(self, index):
        return self.input_gt_imgs_list[index][1]


class CamusResizedDataset(Dataset):
    def __init__(self, input_gt_dir, 
            transforms = A.Compose([
                A.HorizontalFlip(p=.5),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.3, rotate_limit=30, p=0.9),
                ])
            ):
        super().__init__()
        self.transforms = transforms
        self.input_gt_imgs_list = load_input_gt_dir(input_gt_dir)
        print(len(self.input_gt_imgs_list))


    def __len__(self):
        return len(self.input_gt_imgs_list)

    def __getitem__(self, index):
        input_img, gt_img = self.input_gt_imgs_list[index]
        input_img = np.rot90(input_img, 3)
        gt_img = np.where(gt_img>100, 1, 0)
        gt_img = np.rot90(gt_img, 3) # pytorch does not like loading images that has been rotated
        input_img = input_img - np.zeros_like(input_img) # fix for pytorch loading rotated image
        gt_img = gt_img - np.zeros_like(input_img)


        if self.transforms:
            transformed = self.transforms(image=input_img,mask=gt_img)
            input_img = transformed["image"]
            gt_img = transformed["mask"]

        #input_img = np.expand_dims(input_img, 2)
        #print(input_img.shape)
        input_img = np.expand_dims(input_img,0)
        input_img = torch.tensor(input_img, dtype=torch.float32)
        gt_img = torch.tensor(gt_img, dtype=torch.torch.int64)
        #print(gt_img.shape)
        
        return input_img, gt_img

    def get_np_img(self, index):
        return self.input_gt_imgs_list[index][0]

    def get_np_mask(self, index):
        return self.input_gt_imgs_list[index][1]





    





#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, gray_dir, gt_dir, pytorch=True):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, gt_dir) for f in gray_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        
    def combine_files(self, gray_file: Path, gt_dir):
        
        files = {'gray': gray_file, 
                 'gt': gt_dir/gray_file.name.replace('gray', 'gt')}

        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                           ], axis=2)
   
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        #get the image and mask as arrays
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
if __name__ == '__main__':
    input_gt_path_val = 'datasets/TTE/val'
    TTE_val = TTEDataset(input_gt_path_val, True)
    img = TTE_val.get_np_img(10)
    plt.imshow(img)
    plt.show()
    gt = TTE_val.get_np_mask(10)
    plt.imshow(gt)
    plt.show()
    img_t, gt_t = TTE_val[0]
    for i in range(100):
        img = TTE_val.get_np_img(i)
        gt = TTE_val.get_np_mask(i)
        print(img.shape)
        print(gt.shape)

    for i in range(100):
        img,gt = TTE_val[i]
        print(img.shape)
        print(gt.shape)
