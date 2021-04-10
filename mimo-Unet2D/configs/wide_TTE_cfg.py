import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_config():
    config = {
        "model": "wide",
        "epochs": 5,
        "batch_size": 36,
        "learning_rate": 5.0,
        "lr_patience": 1,
        #"input_width": 500,
        #"input_height": 500,
        "channel_ratio": 1,
        
        "dataset": "TTE",
        "isotropic": True,
        "cross_entr_weights": [0.1, 0.3, 0.3, 0.3],

        "train_transforms": A.Compose([
            A.Resize(256,192),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            #ToTensorV2()
            ]),
        "val_transforms": A.Compose([
            A.Resize(256,192),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            #ToTensorV2()
            ]),
    }
    return config






