import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


model = "baseline"
epochs = 25
batch_size = 12
learning_rate = 0.01
lr_patience = 3
channel_ratio = 1.0
dataset = "TTE"
cross_entr_weights = [0.25,0.25,0.25,0.25]
image_size = (352,288)
#image_size = (256, 192)
train_transforms = A.Compose([
            A.Resize(image_size[0],image_size[1]),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255)])

val_transforms = A.Compose([
            A.Resize(image_size[0],image_size[1]),
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








