TRAIN WITH:

python train.py configs/model_config.py
example:
python train.py configs/baseline_TTE_cfg.py

TEST WITH:
need to specify the "random english word" given as the save folder. Look at logdir/other/TTE/model then choose the run to replicate test results with. (models are automatically tested after training, so no need to test again really)

python test.py configs/model_config.py N2_Gesture


HYPERPARAMETER SEARCH:

python hparam_search.py <name-of-study> <number-of-trials>

example:
python hparam_search.py my_latenight_study 20



