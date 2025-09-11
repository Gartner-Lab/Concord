from __future__ import annotations
from typing import Any, Dict, Optional


def ensure_csr(adata: AnnData) -> None:
    """
    Make sure `adata.X` is row-compressed (CSR).
    Converts in place; safe to call multiple times.
    """
    from scipy.sparse import isspmatrix_csr, issparse
    import scipy.sparse as sp
    if issparse(adata.X):
        if not isspmatrix_csr(adata.X):           # CSC → CSR (one copy)
            adata.X = adata.X.tocsr()
    else:                                         # dense → CSR
        adata.X = sp.csr_matrix(adata.X)


def run_scanorama(adata, batch_key="batch", output_key="Scanorama", dimred=100, return_corrected=False):
    import scanorama
    import numpy as np
    ensure_csr(adata)
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
    *,
    # ------------------------------------------------- “fixed” args
    batch_key: str = "batch",
    class_key: Optional[str] = None,
    output_key: str = "Concord",
    mode: str = "default",                # "default" | "decoder" | "class" | "naive"
    seed: int = 42,
    device: str = "cpu",
    n_epochs: int | None = None,  
    save_dir: str | None = None,
    preload_dense: bool = False,
    verbose: bool = False,
    return_corrected: bool = False,
    # ------------------------------------------------- convenience optionals
    latent_dim: int | None = None,
    batch_size: int | None = None,
    input_feature: Optional[list[str]] = None,
    encoder_dims: Optional[list[int]] = None,
    decoder_dims: Optional[list[int]] = None,
    element_mask_prob: float | None = None,
    feature_mask_prob: float | None = None,
    clr_temperature: float | None = None,
    clr_beta: float | None = None,
    p_intra_knn: float | None = None,
    p_intra_domain: float | None = None,
    sampler_emb: str | None = None,
    sampler_knn: int | None = None,
    dropout_prob: float | None = None,
    lr: float | None = None,
    domain_coverage: dict[str, float] | None = None,
    # ------------------------------------------------- NEW: free-form extras
    concord_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Thin wrapper around `Concord`.

    Any keys in `concord_kwargs` are forwarded to the model constructor after
    the standard parameters, so they **override** duplicates.
    """

    from .. import Concord     # local import keeps import-time cost minimal

    # ---------- core (always supplied) -------------------------------------
    kwargs: Dict[str, Any] = dict(
        adata=adata,
        domain_key=batch_key if mode != "naive" else None,
        class_key=class_key if mode == "class" else None,
        use_classifier=(mode == "class"),
        use_decoder=(mode == "decoder"),
        domain_embedding_dim=8,
        seed=seed,
        verbose=verbose,
        device=device,
        save_dir=save_dir,
    )

    # ---------- convenience optionals --------------------------------------
    optional_params = {
        "n_epochs":              n_epochs,
        "latent_dim":            latent_dim,
        "batch_size":            batch_size,
        "input_feature":         input_feature,
        "encoder_dims":          encoder_dims,
        "decoder_dims":          decoder_dims,
        "element_mask_prob":     element_mask_prob,
        "feature_mask_prob":     feature_mask_prob,
        "clr_temperature":       clr_temperature,
        "clr_beta":              clr_beta,
        "p_intra_knn":           p_intra_knn,
        "p_intra_domain":        p_intra_domain,
        "sampler_emb":           sampler_emb,  
        "sampler_knn":           sampler_knn,
        "lr":                    lr,
        "dropout_prob":          dropout_prob,
        "preload_dense": preload_dense,
        "domain_coverage":       domain_coverage,
    }
    kwargs.update({k: v for k, v in optional_params.items() if v is not None})

    # ---------- free-form overrides ----------------------------------------
    if concord_kwargs:
        kwargs.update(concord_kwargs)     # user takes ultimate precedence

    # ---------- run ---------------------------------------------------------
    model = Concord(**kwargs)
    model.fit_transform(output_key=output_key, return_decoded=return_corrected)
    return model            # handy if caller wants the trained model


