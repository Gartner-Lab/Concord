
from __future__ import annotations
import os
import time
import logging
from typing import Any, Callable, Dict, Tuple
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
        import psutil
        """Current resident‑set size (RSS) in **MB**."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 2

    @staticmethod
    def get_peak_ram_mb() -> float:
        """High‑water‑mark RSS (ru_maxrss) in **MB**.

        Good for overall memory footprint monitoring, independent of deltas.
        """
        import resource
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



# -----------------------------------------------------------------------------
# Generic function profiler
# -----------------------------------------------------------------------------

def profiled_run(
    name: str,
    fn: Callable[[], None],
    *,
    profiler: MemoryProfiler,
    logger: logging.Logger | None = None,
    compute_umap: bool = False,
    adata=None,
    output_key: str | None = None,
    umap_params: Dict[str, Any] | None = None,
) -> Tuple[float | None, float | None, float | None]:
    """Execute *fn* and return (time_sec, ΔRAM_MB, peak_VRAM_MB).

    A small helper so other scripts can time/profile any function with the same
    logic we use in the integration pipeline.
    """
    import gc
    if logger is None:
        logger = logging.getLogger(__name__)

    gc.collect()
    ram_before = profiler.get_ram_mb()
    profiler.reset_peak_vram()

    try:
        with Timer() as t:
            fn()
    except Exception as exc:
        logger.error("❌ %s failed: %s", name, exc)
        return None, None, None

    # success ----------------------------------------------------------------
    gc.collect()
    ram_after = profiler.get_ram_mb()
    delta_ram = max(0.0, ram_after - ram_before)
    peak_vram = profiler.get_peak_vram_mb()

    logger.info("%s: %.2fs | %.2f MB RAM | %.2f MB VRAM", name, t.interval, delta_ram, peak_vram)

    # optional UMAP
    if compute_umap and output_key and adata is not None and umap_params is not None:
        from ..utils.dim_reduction import run_umap  # local import avoids heavy deps
        try:
            logger.info("Running UMAP on %s …", output_key)
            run_umap(adata, source_key=output_key, result_key=f"{output_key}_UMAP", **umap_params)
        except Exception as exc:
            logger.error("❌ UMAP for %s failed: %s", output_key, exc)

    return t.interval, delta_ram, peak_vram



def run_and_log(
    method_name: str,
    fn,                      # () -> None
    *,
    adata,
    profiler,
    logger,
    compute_umap: bool,
    output_key: str | None = None,
    umap_params: dict = {},
    time_log: dict = None,
    ram_log: dict = None,
    vram_log: dict = None,
):
    t, dr, pv = profiled_run(
        method_name,
        fn,
        profiler=profiler,
        logger=logger,
        compute_umap=compute_umap,
        adata=adata,
        output_key=output_key,
        umap_params=umap_params,
    )
    if time_log is not None: time_log[method_name] = t
    if ram_log  is not None: ram_log[method_name]  = dr
    if vram_log is not None: vram_log[method_name] = pv
