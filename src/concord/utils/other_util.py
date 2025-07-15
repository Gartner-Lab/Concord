import logging
import random
from pathlib import Path
import numpy as np
import torch


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)




def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def natural_key(string_):
    import re
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def sort_string_list(string_list):
    return sorted(string_list, key=natural_key)




def args_merge(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries left to right, with later dicts overriding earlier ones.
    Accepts any number of dictionaries.
    """
    out = {}
    for d in dicts:
        if d:
            out.update(d)
    return out
