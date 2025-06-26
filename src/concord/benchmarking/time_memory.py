
from __future__ import annotations
import os
import time
import psutil
import resource

try:
    import torch
except ImportError:  # keep import‑time cost low if PyTorch is absent
    torch = None  # type: ignore

# -----------------------------------------------------------------------------
# Timing helper
# -----------------------------------------------------------------------------

class Timer:
    """Context‑manager that records wall‑clock run‑time (high‑resolution)."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc):
        self.interval = time.perf_counter() - self._start  # seconds

    # Make it usable as a simple stopwatch without the *with* block
    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        self.interval = time.perf_counter() - self._start
        return self.interval


# -----------------------------------------------------------------------------
# Memory helper
# -----------------------------------------------------------------------------

class MemoryProfiler:
    """Light‑weight RAM + VRAM (CUDA / MPS) tracker.

    Notes
    -----
    *   `get_ram_mb` returns *current* RSS.
    *   `get_peak_vram_mb` returns peak CUDA/MPS memory **since the last
        `reset_peak_vram`** call, so typical usage is::

            profiler.reset_peak_vram()
            run_heavy_function()
            vram_delta = profiler.get_peak_vram_mb()
    """

    def __init__(self, *, device: str | int | torch.device = "cpu") -> None:  # type: ignore[name‑defined]
        self.device = torch.device(device) if torch is not None else str(device)

    # ----------------------- RAM --------------------------------------------
    @staticmethod
    def get_ram_mb() -> float:
        """Current resident‑set size (RSS) in **MB**."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 2

    @staticmethod
    def get_peak_ram_mb() -> float:
        """High‑water‑mark RSS (ru_maxrss) in **MB**.

        Good for overall memory footprint monitoring, independent of deltas.
        """
        ru_peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux/macOS ru_maxrss is KiB; on Windows it is bytes.
        scale = 1 if os.name == "posix" else 1024
        return ru_peak_kb / (1024 * scale)

    # ----------------------- VRAM -------------------------------------------
    def reset_peak_vram(self) -> None:
        if torch is None:
            return
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and torch.backends.mps.is_available():  # type: ignore[attr‑defined]
            torch.mps.empty_cache()  # best we can do on Apple silicon

    def get_peak_vram_mb(self) -> float:
        if torch is None:
            return 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            return torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        if self.device.type == "mps" and torch.backends.mps.is_available():  # type: ignore[attr‑defined]
            return torch.mps.current_allocated_memory() / 1024 ** 2  # type: ignore[attr‑defined]
        return 0.0

