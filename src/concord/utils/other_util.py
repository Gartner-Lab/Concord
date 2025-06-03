import logging
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import time
from . import io
import os


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





class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start




class MemoryProfiler:
    def __init__(self, device='cpu'):
        self.device = device

    def get_peak_ram(self):
        """Returns current RAM usage of this process in MB."""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

    def get_peak_vram(self):
        """Returns peak VRAM usage in MB for CUDA or MPS, else 0."""
        device = self.device
        try:
            import torch
        except ImportError:
            return 0.0

        if device.startswith('cuda') and hasattr(torch, 'cuda') and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        elif device.startswith('mps') and hasattr(torch, 'mps') and torch.backends.mps.is_available():
            # PyTorch MPS doesn't have max_memory_allocated; can only get total used
            return torch.mps.current_allocated_memory() / 1024**2
        return 0.0

    def reset_peak_vram(self):
        device = self.device
        try:
            import torch
        except ImportError:
            return

        if device.startswith('cuda') and hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elif device.startswith('mps') and hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
