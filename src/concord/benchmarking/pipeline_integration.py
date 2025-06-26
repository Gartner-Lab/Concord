from __future__ import annotations
import gc
import logging
from typing import Callable, Dict, Iterable, List, Optional
import scanpy as sc
import pandas as pd
from .time_memory import Timer, MemoryProfiler


# -----------------------------------------------------------------------------
# Integration benchmarking pipeline (simplified wrap‑up)
# -----------------------------------------------------------------------------

def run_integration_methods_pipeline(
    adata,  # AnnData
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
):
    """Run selected single‑cell integration methods and profile run‑time & memory."""

    # ------------------------------------------------------------------ setup
    logger = logging.getLogger(__name__)
    handler: Optional[logging.Handler]
    if verbose:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(handler)
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

    # ---------------------------------------------------------------- helpers

    def profiled_run(method_name: str, fn: Callable[[], None], output_key: Optional[str] = None):
        """Run *fn* while recording ΔRAM and peak VRAM."""
        gc.collect()
        ram_before = profiler.get_ram_mb()
        profiler.reset_peak_vram()

        try:
            with Timer() as t:
                fn()
        except Exception as exc:
            logger.error("❌ %s failed: %s", method_name, exc)
            time_log[method_name] = ram_log[method_name] = vram_log[method_name] = None
            return

        # success ────────────────────────────────────────────────────────────
        time_log[method_name] = t.interval
        gc.collect()
        ram_after = profiler.get_ram_mb()
        ram_log[method_name] = max(0.0, ram_after - ram_before)
        vram_log[method_name] = profiler.get_peak_vram_mb()

        logger.info("%s: %.2fs | %.2f MB RAM | %.2f MB VRAM", method_name, t.interval, ram_log[method_name], vram_log[method_name])

        # optional UMAP
        if compute_umap and output_key is not None:
            from ..utils.dim_reduction import run_umap  # local import avoids heavy deps if unused
            try:
                logger.info("Running UMAP on %s …", output_key)
                run_umap(adata, source_key=output_key, result_key=f"{output_key}_UMAP", **umap_params)
            except Exception as exc:
                logger.error("❌ UMAP for %s failed: %s", output_key, exc)

    # ---------------------------------------------------------------- method wrappers (local imports)
    from ..utils import (
        run_concord,
        run_scanorama,
        run_liger,
        run_harmony,
        run_scvi,
        run_scanvi,
    )

    # ------------------------------ CONCORD variants ------------------------
    if "concord_knn" in methods:
        profiled_run(
            "concord_knn",
            lambda: run_concord(
                adata,
                batch_key=batch_key,
                output_key="concord_knn",
                latent_dim=latent_dim,
                p_intra_knn=0.3,
                clr_beta=0.0,
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                mode="knn",
            ),
            "concord_knn",
        )

    if "concord_hcl" in methods:
        profiled_run(
            "concord_hcl",
            lambda: run_concord(
                adata,
                batch_key=batch_key,
                clr_beta=1.0,
                p_intra_knn=0.0,
                output_key="concord_hcl",
                latent_dim=latent_dim,
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                mode="hcl",
            ),
            "concord_hcl",
        )

    if "concord_class" in methods:
        profiled_run(
            "concord_class",
            lambda: run_concord(
                adata,
                batch_key=batch_key,
                class_key=class_key,
                output_key="concord_class",
                latent_dim=latent_dim,
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                mode="class",
            ),
            "concord_class",
        )

    if "concord_decoder" in methods:
        profiled_run(
            "concord_decoder",
            lambda: run_concord(
                adata,
                batch_key=batch_key,
                class_key=class_key,
                output_key="concord_decoder",
                latent_dim=latent_dim,
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                mode="decoder",
            ),
            "concord_decoder",
        )

    if "contrastive" in methods:
        profiled_run(
            "contrastive",
            lambda: run_concord(
                adata,
                batch_key=None,
                clr_beta=0.0,
                p_intra_knn=0.0,
                output_key="contrastive",
                latent_dim=latent_dim,
                return_corrected=return_corrected,
                device=device,
                seed=seed,
                verbose=verbose,
                mode="naive",
            ),
            "contrastive",
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
            run_umap(adata, source_key="unintegrated", result_key="unintegrated_UMAP", **umap_params)

    if "scanorama" in methods:
        profiled_run(
            "scanorama",
            lambda: run_scanorama(
                adata,
                batch_key=batch_key,
                output_key="scanorama",
                dimred=latent_dim,
                return_corrected=return_corrected,
            ),
            "scanorama",
        )

    if "liger" in methods:
        profiled_run(
            "liger",
            lambda: run_liger(
                adata,
                batch_key=batch_key,
                count_layer=count_layer,
                output_key="liger",
                k=latent_dim,
                return_corrected=return_corrected,
            ),
            "liger",
        )

    if "harmony" in methods:
        if "X_pca" not in adata.obsm:
            logger.info("Running PCA for harmony …")
            sc.tl.pca(adata, n_comps=latent_dim)
        profiled_run(
            "harmony",
            lambda: run_harmony(
                adata,
                batch_key=batch_key,
                input_key="X_pca",
                output_key="harmony",
                n_comps=latent_dim,
            ),
            "harmony",
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
        profiled_run("scvi", _train_scvi, "scvi")

    if "scanvi" in methods:
        profiled_run(
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
            "scanvi",
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
