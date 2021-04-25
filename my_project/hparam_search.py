import argparse
import torch
import os
import optuna
import albumentations as A
import torch.nn as nn
from train import init_train
from optuna.visualization import *
from plotly.subplots import make_subplots


class HparamStudy:
    def __init__(self, study_name):
        self.study_name = study_name


    def __call__(self, trial):

        torch.cuda.empty_cache()

        metric = 0.0

        im_sz = trial.suggest_categorical("image size", [(352, 288),  (256, 192)])
        #im_sz = (352,288)

        cfg = {}
        cfg["custom_logdir"] = os.path.join(self.study_name, f'imsz{im_sz[0]}x{im_sz[1]}')
        cfg["dataset"] = "TTE"
        cfg["epochs"] = 60
        cfg["image_width"] = im_sz[0]
        cfg["image_height"] = im_sz[1]

        #cr_entr_weights = trial.suggest_categorical("cr_entr_weights", ["equal", "weighted", "heavy_weighted"])
        cr_entr_weights = trial.suggest_categorical("cr_entr_weights", ["weighted", "intuition"])
        if cr_entr_weights == "equal":
            cfg["cross_entr_weights"] = [0.25,0.25,0.25,0.25]
        elif cr_entr_weights == "weighted":
            cfg["cross_entr_weights"] = [0.1,0.3,0.3,0.3]
        elif cr_entr_weights == "heavy_weighted":
            cfg["cross_entr_weights"] = [0.04,0.32,0.32,0.32]
        elif cr_entr_weights == "calculated":
            cfg["cross_entr_weights"] = [0.02857,0.26576,0.2369,0.4688]
        elif cr_entr_weights == "intuition":
            cfg["cross_entr_weights"] = [0.1,0.35,0.2,0.35]




        #im_sz = trial.suggest_categorical("image size", [(512, 384), (256, 192), (384, 512), (192, 256)])



        model = trial.suggest_categorical("model", ["unet_multiinp", "unet_normal"])
        cfg["model"] = model
        
        # HYPERPARAMS #
        batch_size = trial.suggest_categorical("batch_sz", [4, 8])
        cfg["batch_size"] = batch_size

        learning_rate = trial.suggest_float("learning_rate", 1e-3, 3e-3, log=True)
        cfg["learning_rate"] = learning_rate

        cfg["lr_patience"] = 3

        channel_ratio = trial.suggest_categorical("channel_ratio", [2.0, 2.4, 2.6, 2.8])
        cfg["channel_ratio"] = channel_ratio


        # TRANSFORM HP #
        train_transforms = []
        val_transforms = []
        train_transforms.append(A.Resize(im_sz[0],im_sz[1]))
        val_transforms.append(A.Resize(im_sz[0], im_sz[1]))


        #use_blur = trial.suggest_categorical("blur", ["gaussian", "normal", "none", "median"])
        use_blur=False
        if use_blur == "normal":
            train_transforms.append(A.Blur(blur_limit=(5,5), always_apply=True))
            val_transforms.append(A.Blur(blur_limit=(5,5), always_apply=True))
        elif use_blur == "gaussian":
            train_transforms.append(A.GaussianBlur(blur_limit=(5,5), always_apply=True))
            val_transforms.append(A.GaussianBlur(blur_limit=(5,5), always_apply=True))
        elif use_blur == "median":
            train_transforms.append(A.MedianBlur(blur_limit=(5,5), always_apply=True))
            val_transforms.append(A.MedianBlur(blur_limit=(5,5), always_apply=True))



        #use_clahe = trial.suggest_int("clahe", 0, 1)
        use_clahe = False
        if use_clahe:
            train_transforms.append(A.CLAHE (clip_limit=(3.0,3.0), tile_grid_size=(8, 8), always_apply=True))
            val_transforms.append(A.CLAHE (clip_limit=(3.0,3.0), tile_grid_size=(8, 8), always_apply=True))



        use_shift_scale_rotate = trial.suggest_int("shift_sc_rot", 0, 1)
        if use_shift_scale_rotate:
            train_transforms.append(A.ShiftScaleRotate(
                shift_limit=0.0, 
                scale_limit=0.05, 
                rotate_limit=10, p=0.7))

        #use_brightness_contrast = trial.suggest_int("contrast", 0, 1)
        use_brightness_contrast = False
        if use_brightness_contrast:
            train_transforms.append(A.RandomBrightnessContrast (brightness_limit=0.05, contrast_limit=0.05, p=0.6))


        train_transforms.append(A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255))
        val_transforms.append(A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255))
        cfg["train_transforms"] = A.Compose(train_transforms)
        cfg["val_transforms"] = A.Compose(val_transforms)

        for key in cfg:
            print(key, cfg[key])

        try:
            metric = init_train(cfg)
        except RuntimeError as err:
            #print(err.message)
            if ("CUDA" in err.args[0]):
                torch.cuda.empty_cache()
                raise RuntimeError
            else:
                raise RuntimeError

        return metric 




def main():
    print("Main")

    parser = argparse.ArgumentParser(description='Start hparam search, enter study name for the hparam search')
    parser.add_argument('study_name')
    parser.add_argument('num_trials')
    args = parser.parse_args()
    study_name = args.study_name
    n_trials = int(args.num_trials)
    


    study = optuna.create_study(direction='maximize')
    study.optimize(
        HparamStudy(study_name), 
        n_trials=n_trials, 
        catch=(RuntimeError,RuntimeError))

    df = study.trials_dataframe()
    df = df.sort_values("value", ascending=False)
    best_hp = df.head(15)
    print(best_hp)


    study_dir = os.path.join("hparam_search", study_name)
    try:
        os.mkdir(study_dir)
    except: 
        pass

    best_hp.to_csv(os.path.join(study_dir, "best_runs.csv"))

    cont = plot_contour(study)
    hist = plot_optimization_history(study)
    parallel = plot_parallel_coordinate(study)
    importance = plot_param_importances(study)
    slice_pl = plot_slice(study)

    cont.write_image(os.path.join(study_dir, "cont.png"))
    hist.write_image(os.path.join(study_dir, "hist.png"))
    parallel.write_image(os.path.join(study_dir, "parallel.png"))
    importance.write_image(os.path.join(study_dir, "importance.png"))
    slice_pl.write_image(os.path.join(study_dir, "slice.png"))


if __name__ == '__main__':
    main()

