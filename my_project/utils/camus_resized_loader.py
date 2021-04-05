import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def load_tif(img_path):
    im = Image.open(img_path)
    return np.array(im)
    
def load_input_gt_tuple(input_gt_path, patient):
    input_patient_path = os.path.join(input_gt_path, "input", "gray_" + patient)
    input_patient_path += '.tif'
    gt_patient_path = os.path.join(input_gt_path, "gt", "gt_" + patient)
    gt_patient_path += '.tif'
    input_img = load_tif(input_patient_path)
    gt_img = load_tif(gt_patient_path)
    return input_img, gt_img

def load_input_gt_dir(input_gt_path):
    patients = os.listdir(os.path.join(input_gt_path, "input"))
    p = "patient"
    patients = [patient[patient.index(p) : patient.index(p) + len(p) + 4] for patient in patients]
    input_gt_tuples = [load_input_gt_tuple(input_gt_path, patient) for patient in patients]
    return input_gt_tuples
    

    





if __name__ == '__main__':
    input_gt_path = '/home/ola/projects/final-project-TDT4265/Unet2D/datasets/CAMUS_resized/val'
    patient = 'patient0008'
    input_gt_tuples = load_input_gt_dir(input_gt_path)
    for input_img, gt_img in input_gt_tuples:
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(input_img)
        axarr[1].imshow(gt_img)
        plt.show()


