
from __future__ import annotations
import os
import json
from typing import Any, Dict, Optional, Iterable
from .time_memory import MemoryProfiler
from ..utils import args_merge

import pandas as pd
from ..utils import (
    run_pca, run_umap, run_tsne, run_diffusion_map, run_NMF,
    run_SparsePCA, run_FactorAnalysis, run_FastICA, run_LDA,
    run_zifa, run_phate, run_concord, run_scvi
)
from .time_memory import run_and_log
from .. import logger



def run_dimensionality_reduction_pipeline(
    adata,
    source_key: str = "X",
    methods: Iterable[str] = (
        "PCA", "UMAP", "t-SNE", "DiffusionMap", "NMF",
        "SparsePCA", "FactorAnalysis", "FastICA", "LDA",
        "ZIFA", "scVI", "PHATE", "concord", "concord_hcl", "concord_knn"
    ),
    n_components: int = 10,
    seed: int = 42,
    device: str = "cpu",
    save_dir: str = "./",
    concord_kwargs: Optional[Dict[str, Any]] = None,
):

    os.makedirs(save_dir, exist_ok=True)
    ckws = (concord_kwargs or {}).copy()

    time_log: Dict[str, float | None] = {}
    ram_log: Dict[str, float | None] = {}
    vram_log: Dict[str, float | None] = {}

    profiler = MemoryProfiler(device=device)

    def _run(method, fn, output_key=None):
        run_and_log(
            method,
            fn,
            adata=adata,
            profiler=profiler,
            logger=logger,
            compute_umap=False,
            output_key=output_key or method,
            time_log=time_log,
            ram_log=ram_log,
            vram_log=vram_log,
        )

    # Core methods
    if "PCA" in methods:
        _run("PCA", lambda: run_pca(adata, source_key=source_key, result_key='PCA', n_pc=n_components, random_state=seed))

    if "UMAP" in methods:
        _run("UMAP", lambda: run_umap(adata, source_key=source_key, result_key="UMAP", random_state=seed), output_key="UMAP")

    if "t-SNE" in methods:
        _run("t-SNE", lambda: run_tsne(adata, source_key=source_key, result_key="tSNE", random_state=seed), output_key="tSNE")

    if "DiffusionMap" in methods:
        _run("DiffusionMap", lambda: run_diffusion_map(adata, source_key=source_key, n_neighbors=15, n_components=n_components, result_key="DiffusionMap", seed=seed), output_key="DiffusionMap")

    if "NMF" in methods:
        _run("NMF", lambda: run_NMF(adata, source_key=source_key, n_components=n_components, result_key="NMF", seed=seed), output_key="NMF")

    if "SparsePCA" in methods:
        _run("SparsePCA", lambda: run_SparsePCA(adata, source_key=source_key, n_components=n_components, result_key="SparsePCA", seed=seed), output_key="SparsePCA")

    if "FactorAnalysis" in methods:
        _run("FactorAnalysis", lambda: run_FactorAnalysis(adata, source_key=source_key, n_components=n_components, result_key="FactorAnalysis", seed=seed), output_key="FactorAnalysis")

    if "FastICA" in methods:
        _run("FastICA", lambda: run_FastICA(adata, source_key=source_key, result_key="FastICA", n_components=n_components, seed=seed), output_key="FastICA")

    if "LDA" in methods:
        _run("LDA", lambda: run_LDA(adata, source_key=source_key, result_key="LDA", n_components=n_components, seed=seed), output_key="LDA")

    if "ZIFA" in methods:
        _run("ZIFA", lambda: run_zifa(adata, source_key=source_key, log=True, result_key="ZIFA", n_components=n_components), output_key="ZIFA")

    if "scVI" in methods:
        _run("scVI", lambda: run_scvi(adata, batch_key=None, output_key="scVI", return_model=False, return_corrected=False, transform_batch=None), output_key="scVI")

    if "PHATE" in methods:
        _run("PHATE", lambda: run_phate(adata, layer=source_key, n_components=2, result_key="PHATE", seed=seed), output_key="PHATE")


    concord_variants = {
        "concord_knn": {
            "mode": "default",
            "extra_kwargs": {"p_intra_knn": 0.3, "clr_beta": 0.0},
        },
        "concord_hcl": {
            "mode": "default",
            "extra_kwargs": {"p_intra_knn": 0.0, "clr_beta": 1.0},
        },
        "concord_class": {
            "mode": "class",
            "extra_kwargs": {},
        },
        "concord_decoder": {
            "mode": "decoder",
            "extra_kwargs": {},
        },
        "contrastive": {
            "mode": "naive",
            "extra_kwargs": {"p_intra_knn": 0.0, "clr_beta": 0.0},
        },
    }
    
    for method_name, config in concord_variants.items():
        if method_name not in methods:
            logger.warning(f"Skipping {method_name} as it is not in the methods list.")
            continue
        
        logger.info(f"Running {method_name} with configuration: {config}")
        _run(
            method_name,
            lambda m=method_name, cfg=config: run_concord(
                adata,
                batch_key=None,
                output_key=m,
                mode=cfg["mode"],
                device=device,
                seed=seed,
                **args_merge(
                    dict(latent_dim=n_components),
                    cfg["extra_kwargs"],
                    ckws,  # concord_kwargs user supplied
                ),
            ),
            output_key=method_name,
        )

    # ------------------------ Save results ------------------------
    time_log_path = os.path.join(save_dir, "dimensionality_reduction_timelog.json")
    with open(time_log_path, "w") as f:
        json.dump({"time_sec": time_log, "ram_MB": ram_log, "vram_MB": vram_log}, f, indent=4)
    logger.info(f"📝 Time log saved to {time_log_path}")

    return pd.DataFrame({
        "time_sec": pd.Series(time_log),
        "ram_MB": pd.Series(ram_log),
        "vram_MB": pd.Series(vram_log),
    }).sort_index()



