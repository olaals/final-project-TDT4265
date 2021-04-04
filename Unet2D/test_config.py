import sys
sys.path.append("utils")
from file_io import *


args = add_config_parser()
cfg = get_dict(args)
print(cfg)




