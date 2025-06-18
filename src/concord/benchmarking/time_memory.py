
import time
import os

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

        if str(device).startswith('cuda') and hasattr(torch, 'cuda') and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        elif str(device).startswith('mps') and hasattr(torch, 'mps') and torch.backends.mps.is_available():
            # PyTorch MPS doesn't have max_memory_allocated; can only get total used
            return torch.mps.current_allocated_memory() / 1024**2
        return 0.0

    def reset_peak_vram(self):
        device = self.device
        try:
            import torch
        except ImportError:
            return

        if str(device).startswith('cuda') and hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elif str(device).startswith('mps') and hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
