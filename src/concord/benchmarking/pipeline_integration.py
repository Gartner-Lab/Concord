from __future__ import annotations
import logging
from typing import Callable, Dict, Iterable, List, Optional
import scanpy as sc
import pandas as pd

from .time_memory import MemoryProfiler, profiled_run           # ← generic profiler
# local wrappers around the individual methods
from ..utils import (
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

    profiler = MemoryProfiler(device=device)
    time_log: Dict[str, float | None] = {}
    ram_log: Dict[str, float | None] = {}
    vram_log: Dict[str, float | None] = {}

    # helper to run + fill logs ------------------------------------------------
    def _run_and_log(
        method_name: str,
        fn,                      # () -> None
        *,
        output_key: str | None = None,
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
        time_log[method_name] = t
        ram_log[method_name] = dr
        vram_log[method_name] = pv

    ckws = concord_kwargs or {}    # shorthand (empty dict if None)

    # ------------------------------ CONCORD variants ------------------------
    if "concord_knn" in methods:
        _run_and_log(
            "concord_knn",
            lambda: run_concord(
                adata,
                batch_key=batch_key,
                output_key="concord_knn",
                latent_dim=latent_dim,
                p_intra_knn=0.3,               # variant-specific default
                clr_beta=0.0,
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                **ckws,                        # user overrides
            ),
            output_key="concord_knn",
        )

    if "concord_hcl" in methods:
        _run_and_log(
            "concord_hcl",
            lambda: run_concord(
                adata,
                batch_key=batch_key,
                output_key="concord_hcl",
                latent_dim=latent_dim,
                p_intra_knn=0.0,
                clr_beta=1.0,
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                **ckws,
            ),
            output_key="concord_hcl",
        )

    if "concord_class" in methods:
        _run_and_log(
            "concord_class",
            lambda: run_concord(
                adata,
                batch_key=batch_key,
                class_key=class_key,
                output_key="concord_class",
                latent_dim=latent_dim,
                mode="class",
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                **ckws,
            ),
            output_key="concord_class",
        )

    if "concord_decoder" in methods:
        _run_and_log(
            "concord_decoder",
            lambda: run_concord(
                adata,
                batch_key=batch_key,
                class_key=class_key,
                output_key="concord_decoder",
                latent_dim=latent_dim,
                mode="decoder",
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                **ckws,
            ),
            output_key="concord_decoder",
        )

    if "contrastive" in methods:
        _run_and_log(
            "contrastive",
            lambda: run_concord(
                adata,
                batch_key=None,                # “naive” – ignore batch/domain
                output_key="contrastive",
                latent_dim=latent_dim,
                p_intra_knn=0.0,
                clr_beta=0.0,
                mode="naive",
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                **ckws,
            ),
            output_key="contrastive",
        )

    # ------------------------------ baseline methods ------------------------
    if "unintegrated" in methods:
        if "X_pca" not in adata.obsm:
            logger.info("Running PCA for 'unintegrated' embedding …")
            sc.tl.pca(adata, n_comps=latent_dim)
        adata.obsm["unintegrated"] = adata.obsm["X_pca"]
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
        _run_and_log(
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
        _run_and_log(
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
        if "X_pca" not in adata.obsm:
            logger.info("Running PCA for harmony …")
            sc.tl.pca(adata, n_comps=latent_dim)
        _run_and_log(
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

    if "scvi" in methods:
        _run_and_log("scvi", _train_scvi, output_key="scvi")

    if "scanvi" in methods:
        _run_and_log(
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
