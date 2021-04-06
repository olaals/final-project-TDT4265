import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_config():
    config = {
        "model": "baseline",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.01,
        "input_width": 500,
        "input_height": 500,
        "channel_ratio": 1,
        
        "dataset": "TTE",
        "isotropic": True,

        "train_transforms": A.Compose([
            A.Resize(192,256),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            #ToTensorV2()
            ]),
        "val_transforms": A.Compose([
            A.Resize(192,256),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            #ToTensorV2()
            ]),
    }
    return config






