import rawpy
import imageio
import numpy as np
import glob
import os

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def make_isotropic(image, interpolator = sitk.sitkLinear):
    '''
    Resample an image to isotropic pixels (using smallest spacing from original) and save to file. Many file formats
    (jpg, png,...) expect the pixels to be isotropic. By default the function uses a linear interpolator. For
    label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
    '''
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing]*image.GetDimension()
    new_size = [int(round(osz*ospc/min_spacing)) for osz,ospc in zip(original_size, original_spacing)]
    new_size[-1] = original_size[-1]
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                         image.GetPixelID())

def load_itk(filename, isotropic=False, is_segmentation=False):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    if isotropic and not is_segmentation:
        itkimage = make_isotropic(itkimage)
    if isotropic and is_segmentation:
        itkimage = make_isotropic(itkimage, sitk.sitkNearestNeighbor)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    #origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    #spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return np.squeeze(ct_scan)

def load_train_and_gt(path, isotropic=False):
    train_path = path + ".mhd"
    gt_path = path + "_gt.mhd"
    train_img = load_itk(train_path, isotropic)
    gt_img = load_itk(gt_path, isotropic, is_segmentation=True)
    return train_img, gt_img

def load_patient_dir(patient_dir, isotropic=False):
    mhd_files = [name for name in os.listdir(patient_dir) if name.endswith("ED.mhd") or name.endswith("ES.mhd")]
    mhd_paths = [os.path.join(patient_dir, os.path.splitext(mhd_file)[0]) for mhd_file in mhd_files]
    img_train_gt_list = []
    for mhd_path in mhd_paths:
        train_gt_tup = load_train_and_gt(mhd_path, isotropic)
        img_train_gt_list.append(train_gt_tup)
    return img_train_gt_list

def plot_train_gt_imgs(train_img, gt_img):
    f,axarr = plt.subplots(1,2)
    axarr[0].imshow(train_img)
    im = axarr[1].imshow(gt_img)
    #plt.colorbar(im, ax=axarr[1])
    plt.show()

def get_all_patients_imgs(path, isotropic=False):
    train_gt_list = []
    patient_paths = [os.path.join(path,patient_dir) for patient_dir in os.listdir(path)]
    for patient_path in patient_paths:
        train_gt_list += load_patient_dir(patient_path, isotropic)
    return train_gt_list


