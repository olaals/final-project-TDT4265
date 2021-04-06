import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_config():
    config = {
        "model": "improved_v2",
        "epochs": 20,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "input_width": 500,
        "input_height": 500,
        
        "dataset": "CAMUS_resized",
        "train_dir": "datasets/CAMUS_resized/train",
        "val_dir": "datasets/CAMUS_resized/val",
        "test_dir": "datasets/CAMUS_resized/test",

        "train_transforms": A.Compose([
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=0.7)
            ]),
        "val_transforms": A.Compose([
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            ]),
    }
    return config






