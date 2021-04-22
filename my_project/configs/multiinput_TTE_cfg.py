import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


model = "multiinput"
epochs = 30
batch_size = 8
learning_rate = 0.001
lr_patience = 3
channel_ratio = 2.4
dataset = "TTE"
cross_entr_weights = [0.1,0.3,0.3,0.3]
image_size = (352,288)
#image_size = (128, 96)
train_transforms = A.Compose([
            A.Resize(image_size[0],image_size[1]),
            #A.CLAHE (clip_limit=(3.0,3.0), tile_grid_size=(8, 8), always_apply=True),
            #A.Blur(blur_limit=(5,5), always_apply=True),
            #A.RandomBrightnessContrast (brightness_limit=0.0, contrast_limit=(0.25, 0.25), always_apply=True),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=0.15,
                rotate_limit=10, 
                p=0.7, border_mode=0)
            ])

val_transforms = A.Compose([
            A.Resize(image_size[0],image_size[1]),
            #A.CLAHE (clip_limit=(3.0,3.0), tile_grid_size=(8, 8), always_apply=True),
            #A.Blur(blur_limit=(5,5), always_apply=True),
            #A.RandomBrightnessContrast (brightness_limit=0.0, contrast_limit=(0.25,0.25), always_apply=True),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            ])




def get_config():
    config = {
        "model": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "learning_rate": learning_rate,
        "lr_patience": lr_patience,
        "channel_ratio": channel_ratio,
        "dataset": dataset,
        "cross_entr_weights": cross_entr_weights,
        "train_transforms": train_transforms,
        "val_transforms": val_transforms,
    }
    return config








