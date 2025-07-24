from __future__ import annotations
import logging
from typing import Callable, Dict, Iterable, List, Optional
from ..utils import args_merge
from .time_memory import MemoryProfiler
import scanpy as sc
import pandas as pd

from .time_memory import run_and_log
# local wrappers around the individual methods
from ..utils.integration_methods import (
    run_concord,
    run_scanorama,
    run_liger,
    run_harmony,
    run_scvi,
    run_scanvi,
)

# -----------------------------------------------------------------------------
# Integration benchmarking pipeline (simplified wrap‑up)
# -----------------------------------------------------------------------------



def expand_one_at_a_time(base: dict, grid: dict, base_tag: str = "concord") -> List[dict]:
    """
    Return a list of concord_kwargs dicts.
    Each dict == base plus ONE (key, value) from grid,
    and includes a unique 'tag' + 'output_key'.
    """
    import copy
    jobs = []
    for param, values in grid.items():
        for v in values:
            kw                    = copy.deepcopy(base)
            kw[param]             = v
            if param == "input_feature":
                tag = f'{param}-{len(v)}'  # e.g. "input_feature_gene"
            elif param == "encoder_dims":
                tag = f"{param}-{v[0]}"
            else:
                tag                   = f"{param}-{v}"
            kw["output_key"]      = f"{base_tag}_{tag}"   # you can template this
            jobs.append(kw)
    return jobs


def expand_product(base: dict,
                   grid: dict,
                   joint_keys: tuple,
                   base_tag: str = "concord") -> list[dict]:
    """
    Cartesian-product expansion for the parameters in `joint_keys`.
    All other keys are varied one-at-a-time (like before).
    """
    import copy
    from itertools import product
    jobs = []

    # 1) parameters to vary jointly
    pvals = [grid[k] for k in joint_keys]
    for combo in product(*pvals):
        kw  = copy.deepcopy(base)
        tag = []
        for k, v in zip(joint_keys, combo):
            kw[k] = v
            tag.append(f"{k}-{v}")
        kw["output_key"] = f"{base_tag}_{'_'.join(tag)}"
        jobs.append(kw)

    # 2) all remaining one-at-a-time params
    for k, values in grid.items():
        if k in joint_keys:
            continue
        for v in values:
            kw               = copy.deepcopy(base)
            kw[k]            = v
            suffix           = f"{k}-{v if k!='encoder_dims' else v[0]}"
            kw["output_key"] = f"{base_tag}_{suffix}"
            jobs.append(kw)

    return jobs




def run_integration_methods_pipeline(
    adata,                                   # AnnData
    methods: Optional[Iterable[str]] = None,
    *,
    batch_key: str = "batch",
    count_layer: str = "counts",
    class_key: str = "cell_type",
    latent_dim: int = 30,
    device: str = "cpu",
    return_corrected: bool = False,
    transform_batch: Optional[List[str]] = None,
    compute_umap: bool = False,
    umap_n_components: int = 2,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
    # NEW: user-supplied Concord kwargs
    concord_kwargs: Optional[Dict[str, Any]] = None,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    
    """Run selected single‑cell integration methods and profile run‑time & memory."""

    # ------------------------------------------------------------------ setup
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(logging.Formatter("%(message)s"))
            _h.setLevel(logging.INFO)
            logger.addHandler(_h)
    else:
        logger.setLevel(logging.ERROR)


    if methods is None:
        methods = [
            "unintegrated",
            "scanorama",
            "liger",
            "harmony",
            "scvi",
            "scanvi",
            "concord_knn",
            "concord_hcl",
            "concord_class",
            "concord_decoder",
            "contrastive",
        ]

    # UMAP parameters (re‑used)
    umap_params = dict(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=seed,
    )

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
            compute_umap=compute_umap,
            output_key=output_key or method,
            time_log=time_log,
            ram_log=ram_log,
            vram_log=vram_log,
        )

    ckws = (concord_kwargs or {}).copy()
    out_key = ckws.pop("output_key", None)

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
            continue

        _run(
            method_name,
            lambda m=method_name, cfg=config: run_concord(
                adata,
                batch_key=batch_key if m != "contrastive" else None,
                class_key=class_key,
                output_key=out_key or m,
                mode=cfg["mode"],
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                **args_merge(
                    dict(latent_dim=latent_dim, verbose=verbose),
                    cfg["extra_kwargs"],
                    ckws,  # concord_kwargs user supplied
                ),
            ),
            output_key=out_key or method_name,
        )
        
    # ------------------------------ baseline methods ------------------------
    if "unintegrated" in methods:
        if "X_pca" not in adata.obsm or adata.obsm["X_pca"].shape[1] < latent_dim:
            logger.info("Running PCA for 'unintegrated' embedding …")
            sc.tl.pca(adata, n_comps=latent_dim)

        # Only take the latent_dim components and store them in "unintegrated"
        adata.obsm["unintegrated"] = adata.obsm["X_pca"][:, :latent_dim].copy()
        
        if compute_umap:
            from ..utils.dim_reduction import run_umap
            logger.info("Running UMAP on unintegrated …")
            run_umap(
                adata,
                source_key="unintegrated",
                result_key="unintegrated_UMAP",
                **umap_params,
            )

    if "scanorama" in methods:
        _run(
            "scanorama",
            lambda: run_scanorama(
                adata,
                batch_key=batch_key,
                output_key="scanorama",
                dimred=latent_dim,
                return_corrected=return_corrected,
            ),
            output_key="scanorama",
        )

    if "liger" in methods:
        _run(
            "liger",
            lambda: run_liger(
                adata,
                batch_key=batch_key,
                count_layer=count_layer,
                output_key="liger",
                k=latent_dim,
                return_corrected=return_corrected,
            ),
            output_key="liger",
        )

    if "harmony" in methods:
        if "X_pca" not in adata.obsm or adata.obsm["X_pca"].shape[1] < latent_dim:
            logger.info("Running PCA for harmony …")
            sc.tl.pca(adata, n_comps=latent_dim)
        _run(
            "harmony",
            lambda: run_harmony(
                adata,
                batch_key=batch_key,
                input_key="X_pca",
                output_key="harmony",
                n_comps=latent_dim,
            ),
            output_key="harmony",
        )

    # ------------------------------ scVI / scANVI ---------------------------
    scvi_model = None

    def _train_scvi():
        nonlocal scvi_model
        scvi_model = run_scvi(
            adata,
            batch_key=batch_key,
            output_key="scvi",
            n_latent=latent_dim,
            return_corrected=return_corrected,
            transform_batch=transform_batch,
            return_model=True,
        )
        if save_dir:
            from pathlib import Path
            save_path = Path(save_dir) / "scvi_model.pt"
            scvi_model.save(save_path, overwrite=True)
            logger.info(f"Saved scVI model to {save_path}")

    if "scvi" in methods:
        _run("scvi", _train_scvi, output_key="scvi")

    if "scanvi" in methods:
        _run(
            "scanvi",
            lambda: run_scanvi(
                adata,
                scvi_model=scvi_model,
                batch_key=batch_key,
                labels_key=class_key,
                output_key="scanvi",
                return_corrected=return_corrected,
                transform_batch=transform_batch,
            ),
            output_key="scanvi",
        )

    # ---------------------------------------------------------------- finish
    logger.info("✅ All selected methods completed.")

    # assemble results table --------------------------------------------------
    df = pd.concat(
        {
            "time_sec": pd.Series(time_log),
            "ram_MB": pd.Series(ram_log),
            "vram_MB": pd.Series(vram_log),
        },
        axis=1,
    ).sort_index()
    return df
