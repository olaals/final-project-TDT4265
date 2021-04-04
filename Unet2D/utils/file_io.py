import os
from importlib import import_module
import sys
sys.path.append("configs")
import argparse

def add_config_parser():
    description = "Train script for pytorch"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config', help="Path to configuration py file")
    args = parser.parse_args()
    return args

def get_dict(args):
    config_path = args.config
    py_config_file = os.path.splitext(os.path.split(config_path)[-1])[0]
    config_import = import_module(py_config_file)
    cfg = config_import.get_config()
    return cfg

    
    

    







