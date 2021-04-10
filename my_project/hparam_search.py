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

        im_sz = (256,192)
        #im_sz = trial.suggest_categorical("image size", [(512, 384), (256, 192), (384, 512), (192, 256)])

        cfg = {}
        cfg["custom_logdir"] = os.path.join(self.study_name, f'imsz{im_sz[0]}x{im_sz[1]}')
        cfg["dataset"] = "TTE"
        cfg["epochs"] = 0

        cr_entr_weights = trial.suggest_categorical("cr_entr_weights", ["equal", "weighted", "heavy_weighted"])
        if cr_entr_weights == "equal":
            cfg["cross_entr_weights"] = [0.25,0.25,0.25,0.25]
        elif cr_entr_weights == "weighted":
            cfg["cross_entr_weights"] = [0.1,0.3,0.3,0.3]
        elif cr_entr_weights == "heavy_weighted":
            cfg["cross_entr_weights"] = [0.04,0.32,0.32,0.32]


        #im_sz = trial.suggest_categorical("image size", [(512, 384), (256, 192), (384, 512), (192, 256)])


        cfg["val_transforms"] = A.Compose([
            A.Resize(im_sz[0],im_sz[1]),
            A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
            ])


        model = trial.suggest_categorical("model", ["baseline", "wide", "longer"])
        cfg["model"] = model
        
        # HYPERPARAMS #
        batch_size = trial.suggest_categorical("batch_sz", [4, 8,16])
        cfg["batch_size"] = batch_size

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        cfg["learning_rate"] = learning_rate
        lr_patience = trial.suggest_int("lr_patience", 3, 10)
        cfg["lr_patience"] = lr_patience

        channel_ratio = trial.suggest_categorical("channel_ratio", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.6, 3.0])
        cfg["channel_ratio"] = channel_ratio


        # TRAIN TRANSFORM HP #
        train_transforms = []

        train_transforms.append(A.Resize(im_sz[0],im_sz[1]))
        train_transforms.append(A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255))

        use_shift_scale_rotate = trial.suggest_int("shift_sc_rot", 0, 1)
        if use_shift_scale_rotate:
            cfg["trfm_rot_scale"] = True
            train_transforms.append(A.ShiftScaleRotate(
                shift_limit=0.0, 
                scale_limit=0.2, 
                rotate_limit=15, p=0.7))
        else:
            cfg["trfm_rot_scale"] = False

        use_blur = trial.suggest_int("blur", 0, 1)
        if use_blur:
            cfg["trfm_blur"] = True
            train_transforms.append(A.Blur(blur_limit=5, always_apply=False, p=0.5))
        else:
            cfg["trfm_blur"] = False


        use_brightness_contrast = trial.suggest_int("contrast", 0, 1)
        if use_brightness_contrast:
            cfg["trfm_contr"] = True
            train_transforms.append(A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, p=0.3))
        else:
            cfg["trfm_contr"] = False


        cfg["train_transforms"] = A.Compose(train_transforms)

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

