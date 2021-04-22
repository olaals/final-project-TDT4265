from mhd_loader import *
import cv2
import  matplotlib.pyplot as plt
import numpy as np

path = "../datasets/TTE/train"
aa = get_all_patients_imgs(path, isotropic=False)
raw, gt = aa[0]
raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
print(raw.shape)

fig,ax = plt.subplots(1,4)

raw_dn = cv2.fastNlMeansDenoisingColored(raw,None,10,10,7,21) 
raw_dn = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img  =clahe.apply(raw_dn)

ax[0].imshow(raw)
ax[1].imshow(raw_dn)
ax[2].imshow(clahe_img)
ax[3].imshow(gt)
plt.show()
