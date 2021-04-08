import os
import optuna
import albumentations as A
import torch.nn as nn
from train import init_train
from optuna.visualization import *
from plotly.subplots import make_subplots


HPARAM_STUDY_NAME = "First study"


def objective(trial):
    metric = 0.0
    print("objective")
    metric += trial.suggest_uniform('uniform1', 1, 3)
    metric += trial.suggest_uniform('uniform2', 1, 3)
    metric += trial.suggest_categorical('cat1', [1,2,3])
    
    return metric

def hparam_study(trial):
    metric = 0.0

    

    cfg = {}
    cfg["custom_logdir"] = HPARAM_STUDY_NAME
    cfg["dataset"] = "TTE"
    cfg["epochs"] = 10
    cfg["model"] = "baseline"
    cfg["val_transforms"] = A.Compose([
        A.Resize(512,384),
        A.Normalize(mean=[0.0],std=[1.0], max_pixel_value=255),
        ])



    batch_size = trial.suggest_categorical("batch_sz", [2,4,8,16])
    cfg["batch_size"] = batch_size

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    cfg["learning_rate"] = learning_rate









    metric = batch_size + learning_rate





    for key in cfg:
        print(key, cfg[key])

    return metric



def main():
    print("Main")


    study = optuna.create_study(direction='maximize')
    study.optimize(hparam_study, n_trials=5)

    df = study.trials_dataframe()
    df = df.sort_values("value", ascending=False)
    best_hp = df.head(15)
    print(best_hp)


    study_dir = os.path.join("hparam_search", HPARAM_STUDY_NAME)
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

