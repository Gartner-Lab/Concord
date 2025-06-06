
def run_scanorama(adata, batch_key="batch", output_key="Scanorama", dimred=100, return_corrected=False):
    import scanorama
    import numpy as np

    batch_cats = adata.obs[batch_key].cat.categories
    adata_list = [adata[adata.obs[batch_key] == b].copy() for b in batch_cats]

    scanorama.integrate_scanpy(adata_list, dimred=dimred)
    adata.obsm[output_key] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))

    if return_corrected:
        corrected = scanorama.correct_scanpy(adata_list)
        adata.layers[output_key + "_corrected"] = np.zeros(adata.shape)
    for i, b in enumerate(batch_cats):
        adata.obsm[output_key][adata.obs[batch_key] == b] = adata_list[i].obsm["X_scanorama"]
        if return_corrected:
            adata.layers[output_key + "_corrected"][adata.obs[batch_key] == b] = corrected[i].X.toarray()



def run_liger(adata, batch_key="batch", count_layer="counts", output_key="LIGER", k=30, return_corrected=False):
    import numpy as np
    import pyliger
    from scipy.sparse import csr_matrix

    bdata = adata.copy()
    # Ensure batch_key is a categorical variable
    if not bdata.obs[batch_key].dtype.name == "category":
        bdata.obs[batch_key] = bdata.obs[batch_key].astype("category")
    batch_cats = bdata.obs[batch_key].cat.categories

    # Set the count layer as the primary data for normalization in Pyliger    
    bdata.X = bdata.layers[count_layer]
    # Convert to csr matrix if not
    if not isinstance(bdata.X, csr_matrix):
        bdata.X = csr_matrix(bdata.X)
    
    # Create a list of adata objects, one per batch
    adata_list = [bdata[bdata.obs[batch_key] == b].copy() for b in batch_cats]
    for i, ad in enumerate(adata_list):
        ad.uns["sample_name"] = batch_cats[i]
        ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)  # Ensures same genes are used in each adata

    # Create a LIGER object from the list of adata per batch
    liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
    liger_data.var_genes = bdata.var_names  # Set genes for LIGER data consistency

    # Run LIGER integration steps
    pyliger.normalize(liger_data)
    pyliger.scale_not_center(liger_data)
    pyliger.optimize_ALS(liger_data, k=k)
    pyliger.quantile_norm(liger_data)


    # Initialize the obsm field for the integrated data
    adata.obsm[output_key] = np.zeros((adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
    
    # Populate the integrated embeddings back into the main AnnData object
    for i, b in enumerate(batch_cats):
        adata.obsm[output_key][adata.obs[batch_key] == b] = liger_data.adata_list[i].obsm["H_norm"]

    if return_corrected:
        corrected_expression = np.zeros(adata.shape)
        for i, b in enumerate(batch_cats):
            H = liger_data.adata_list[i].obsm["H_norm"]  # Latent representation (cells x factors)
            W = liger_data.W  # Gene loadings (genes x factors)
            corrected_expression[adata.obs[batch_key] == b] = H @ W.T

        adata.layers[output_key + "_corrected"] = corrected_expression
    

def run_harmony(adata, batch_key="batch", output_key="Harmony", input_key="X_pca", n_comps=None):
    from harmony import harmonize
    if input_key not in adata.obsm:
        raise ValueError(f"Input key '{input_key}' not found in adata.obsm")
    
    # Check if input_key obsm have enough components
    if n_comps is None:
        n_comps = adata.obsm[input_key].shape[1]
    else:
        if adata.obsm[input_key].shape[1] < n_comps:
            raise ValueError(f"Input key '{input_key}' must have at least {n_comps} components for Harmony integration.")
    
    # Subset the input data to the specified number of components
    input_data = adata.obsm[input_key][:, :n_comps]

    adata.obsm[output_key] = harmonize(input_data, adata.obs, batch_key=batch_key)



def run_scvi(adata, layer="counts", batch_key="batch", gene_likelihood="nb", n_layers=2, n_latent=30, output_key="scVI", return_model=True, return_corrected=False, transform_batch=None):
    import scvi
    # Set up the AnnData object for SCVI
    scvi.model.SCVI.setup_anndata(adata, layer=layer, batch_key=batch_key)
    
    # Initialize and train the SCVI model
    vae = scvi.model.SCVI(adata, gene_likelihood=gene_likelihood, n_layers=n_layers, n_latent=n_latent)
    vae.train()
    
    # Store the latent representation in the specified obsm key
    adata.obsm[output_key] = vae.get_latent_representation()

    if return_corrected:
        corrected_expression = vae.get_normalized_expression(transform_batch=transform_batch)
        adata.layers[output_key + "_corrected" + ("" if transform_batch is None else f"_{transform_batch}")] = corrected_expression
    
    if return_model:
        return vae
    

def run_scanvi(adata, scvi_model=None, layer="counts", batch_key="batch", labels_key="cell_type", unlabeled_category="Unknown", output_key="scANVI", 
               gene_likelihood="nb", n_layers=2, n_latent=30, return_corrected=False, transform_batch=None):
    import scvi
    # Train SCVI model if not supplied
    if scvi_model is None:
        scvi_model = run_scvi(adata, layer=layer, batch_key=batch_key, gene_likelihood=gene_likelihood,
                              n_layers=n_layers, n_latent=n_latent, output_key="scVI", return_model=True)
    
    # Set up and train the SCANVI model
    lvae = scvi.model.SCANVI.from_scvi_model(scvi_model, adata=adata, labels_key=labels_key, unlabeled_category=unlabeled_category)
    lvae.train(max_epochs=20, n_samples_per_label=100)
    
    if return_corrected:
        corrected_expression = lvae.get_normalized_expression(transform_batch=transform_batch)
        adata.layers[output_key + "_corrected" + ("" if transform_batch is None else f"_{transform_batch}")] = corrected_expression
    # Store the SCANVI latent representation in the specified obsm key
    adata.obsm[output_key] = lvae.get_latent_representation()
    

def run_concord(
    adata,
    layer="X",
    preprocess=False,
    batch_key="batch",
    class_key=None,
    output_key="Concord",
    latent_dim=30,
    return_corrected=False,
    seed=42,
    device="cpu",
    save_dir=None,
    mode="default",  # Options: "default", "decoder", "class", "naive"
    n_epochs=10,
    batch_size=64,
    min_p_intra_domain=None,
):
    from ..concord import Concord

    kwargs = {
        "adata": adata,
        "input_feature": None,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "domain_key": batch_key if mode != "naive" else None,
        "class_key": class_key if mode == "class" else None,
        "use_classifier": mode == "class",
        "use_decoder": mode == "decoder",
        "domain_embedding_dim": 8,
        "min_p_intra_domain": min_p_intra_domain,
        "seed": seed,
        "verbose": False,
        "device": device,
        "save_dir": save_dir,
    }

    model = Concord(**kwargs)
    model.encode_adata(input_layer_key=layer, output_key=output_key, preprocess=preprocess, return_decoded=return_corrected)


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
    from .other_util import Timer, MemoryProfiler

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
            from .run_dim_reduction import run_umap
            try:
                logger.info(f"Running UMAP on {output_key}...")
                run_umap(adata, source_key=output_key, result_key=f"{output_key}_UMAP", **umap_params)
            except Exception as e:
                logger.error(f"❌ UMAP for {output_key} failed: {e}")

    # Concord default
    if "concord" in methods:
        profiled_run("concord", lambda: run_concord(
            adata, layer='X', preprocess=True, batch_key=batch_key,
            output_key="concord", latent_dim=latent_dim,
            return_corrected=return_corrected, device=device, seed=seed, mode="default"), "concord")

    # Concord class
    if "concord_class" in methods:
        profiled_run("concord_class", lambda: run_concord(
            adata, layer='X', preprocess=False, batch_key=batch_key,
            class_key=class_key, output_key="concord_class",
            latent_dim=latent_dim, return_corrected=return_corrected, device=device,
            seed=seed, min_p_intra_domain=0.7, mode="class"), "concord_class")

    # Concord decoder
    if "concord_decoder" in methods:
        profiled_run("concord_decoder", lambda: run_concord(
            adata, layer='X', preprocess=False, batch_key=batch_key,
            class_key=class_key, output_key="concord_decoder",
            latent_dim=latent_dim, return_corrected=return_corrected, device=device,
            seed=seed, min_p_intra_domain=0.7, mode="decoder"), "concord_decoder")

    # Contrastive naive
    if "contrastive" in methods:
        profiled_run("contrastive", lambda: run_concord(
            adata, layer='X', preprocess=False, batch_key=None,
            output_key="contrastive", latent_dim=latent_dim,
            return_corrected=return_corrected, device=device, seed=seed, mode="naive"), "contrastive")
        
    # Unintegrated
    if "unintegrated" in methods:
        if "X_pca" not in adata.obsm:
            logger.info("Running PCA to compute 'unintegrated' embedding...")
            sc.tl.pca(adata, n_comps=latent_dim)
        adata.obsm["unintegrated"] = adata.obsm["X_pca"]
        if compute_umap:
            from .run_dim_reduction import run_umap
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
