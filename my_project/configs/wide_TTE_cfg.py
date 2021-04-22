import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_config():
    config = {
        "model": "wide",
        "epochs": 25,
        "batch_size": 8,
        "learning_rate": 0.001,
        "lr_patience": 3,
        "channel_ratio": 2.0,
        
        "dataset": "TTE",
        "cross_entr_weights": [0.1, 0.3, 0.3, 0.3],

        "train_transforms": A.Compose([
            A.Resize(352,288),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            #A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=0.2,
                rotate_limit=15, p=0.7)
            #ToTensorV2()
            ]),
        "val_transforms": A.Compose([
            A.Resize(352,288),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            #ToTensorV2()
            ]),
    }
    return config






