from .time_memory import Timer, MemoryProfiler

def run_integration_methods_pipeline(
    adata,
    methods=None,
    batch_key="batch",
    count_layer="counts",
    class_key="cell_type",
    latent_dim=30,
    device="cpu",
    return_corrected=False,
    transform_batch=None,
    compute_umap=False,
    umap_n_components=2,
    umap_n_neighbors=30,
    umap_min_dist=0.1,
    seed=42,
    verbose=True,
):
    import logging
    import scanpy as sc
    from ..utils import run_concord, run_scanorama, run_liger, run_harmony, run_scvi, run_scanvi


    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    if methods is None:
        methods = [
            "unintegrated",
            "scanorama", "liger", "harmony",
            "scvi", "scanvi",
            "concord", "concord_class", "concord_decoder", "contrastive"
        ]

    umap_params = dict(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=seed,
    )

    profiler = MemoryProfiler(device=device)
    time_log = {}
    ram_log = {}
    vram_log = {}

    timer = Timer()

    def profiled_run(method_name, func, output_key=None):
        # Record RAM/VRAM before
        ram_before = profiler.get_peak_ram()
        profiler.reset_peak_vram()
        vram_before = profiler.get_peak_vram()

        try:
            with timer:
                func()
            time_log[method_name] = timer.interval
            logger.info(f"{method_name} completed in {timer.interval:.2f} sec.")
        except Exception as e:
            logger.error(f"❌ {method_name} failed: {e}")
            time_log[method_name] = None
            ram_log[method_name] = None
            vram_log[method_name] = None
            return

        # RAM/VRAM after
        ram_after = profiler.get_peak_ram()
        vram_after = profiler.get_peak_vram()

        ram_log[method_name] = max(0, ram_after - ram_before)
        vram_log[method_name] = max(0, vram_after - vram_before)

        # Run UMAP separately, not part of the profiling
        if compute_umap and output_key is not None:
            from ..utils.dim_reduction import run_umap
            try:
                logger.info(f"Running UMAP on {output_key}...")
                run_umap(adata, source_key=output_key, result_key=f"{output_key}_UMAP", **umap_params)
            except Exception as e:
                logger.error(f"❌ UMAP for {output_key} failed: {e}")

    # Concord default
    if "concord" in methods:
        profiled_run("concord", lambda: run_concord(
            adata, batch_key=batch_key,
            output_key="concord", latent_dim=latent_dim,
            return_corrected=return_corrected, device=device, seed=seed, mode="default"), "concord")

    # Concord class
    if "concord_class" in methods:
        profiled_run("concord_class", lambda: run_concord(
            adata, batch_key=batch_key,
            class_key=class_key, output_key="concord_class",
            latent_dim=latent_dim, return_corrected=return_corrected, device=device,
            seed=seed, mode="class"), "concord_class")

    # Concord decoder
    if "concord_decoder" in methods:
        profiled_run("concord_decoder", lambda: run_concord(
            adata, batch_key=batch_key,
            class_key=class_key, output_key="concord_decoder",
            latent_dim=latent_dim, return_corrected=return_corrected, device=device,
            seed=seed, mode="decoder"), "concord_decoder")

    # Contrastive naive
    if "contrastive" in methods:
        profiled_run("contrastive", lambda: run_concord(
            adata, batch_key=None,
            output_key="contrastive", latent_dim=latent_dim,
            return_corrected=return_corrected, device=device, seed=seed, mode="naive"), "contrastive")
        
    # Unintegrated
    if "unintegrated" in methods:
        if "X_pca" not in adata.obsm:
            logger.info("Running PCA to compute 'unintegrated' embedding...")
            sc.tl.pca(adata, n_comps=latent_dim)
        adata.obsm["unintegrated"] = adata.obsm["X_pca"]
        if compute_umap:
            from ..utils.dim_reduction import run_umap
            logger.info("Running UMAP on unintegrated...")
            run_umap(adata, source_key="unintegrated", result_key="unintegrated_UMAP", **umap_params)

    # Scanorama
    if "scanorama" in methods:
        profiled_run("scanorama", lambda: run_scanorama(
            adata, batch_key=batch_key, output_key="scanorama",
            dimred=latent_dim, return_corrected=return_corrected), "scanorama")

    # LIGER
    if "liger" in methods:
        profiled_run("liger", lambda: run_liger(
            adata, batch_key=batch_key, count_layer=count_layer,
            output_key="liger", k=latent_dim, return_corrected=return_corrected), "liger")

    # Harmony
    if "harmony" in methods:
        if "X_pca" not in adata.obsm:
            logger.info("Running PCA for harmony...")
            sc.tl.pca(adata, n_comps=latent_dim)
        profiled_run("harmony", lambda: run_harmony(
            adata, batch_key=batch_key, input_key="X_pca",
            output_key="harmony", n_comps=latent_dim), "harmony")

    # scVI
    scvi_model = None
    def _store_scvi_model():
        nonlocal scvi_model
        scvi_model = run_scvi(
            adata, batch_key=batch_key,
            output_key="scvi", n_latent=latent_dim,
            return_corrected=return_corrected, transform_batch=transform_batch,
            return_model=True)

    if "scvi" in methods:
        profiled_run("scvi", _store_scvi_model, "scvi")

    # scANVI
    if "scanvi" in methods:
        profiled_run("scanvi", lambda: run_scanvi(
            adata, scvi_model=scvi_model, batch_key=batch_key,
            labels_key=class_key, output_key="scanvi",
            return_corrected=return_corrected, transform_batch=transform_batch), "scanvi")



    logger.info("✅ Selected methods completed.")
    return time_log, ram_log, vram_log
