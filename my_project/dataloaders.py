import sys
sys.path.append("utils")
from camus_resized_loader import load_input_gt_dir
import numpy as np
import torch
import albumentations as A

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import matplotlib.pyplot as plt


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
    input_gt_path_val = 'datasets/CAMUS_resized/val'
    CAMUS_resized_val = CamusResizedDataset(input_gt_path_val)
    for i in range(len(CAMUS_resized_val)):
        print(i)
        f, axarr = plt.subplots(1,2)
        img,gt = CAMUS_resized_val[i]
        axarr[0].imshow(img)
        axarr[1].imshow(gt)
        plt.show()


    pass

