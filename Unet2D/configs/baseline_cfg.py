import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_config():
    config = {
        "model": "baseline",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 8,
        "input_width": 500,
        "input_height": 500,
        
        "train_dir": "datasets/CAMUS_resized/train",
        "val_dir": "datasets/CAMUS_resized/val",
        "test_dir": "datasets/CAMUS_resized/test",

        "train_transforms": A.Compose([
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            ToTensorV2()
            ]),
        "val_transforms": A.Compose([
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            ToTensorV2()
            ]),
    }
    return config






