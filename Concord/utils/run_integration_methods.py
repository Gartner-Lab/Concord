

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
    

def run_harmony(adata, batch_key="batch", output_key="Harmony", input_key="X_pca"):
    from harmony import harmonize
    if input_key not in adata.obsm:
        raise ValueError(f"Input key '{input_key}' not found in adata.obsm")
    
    adata.obsm[output_key] = harmonize(adata.obsm[input_key], adata.obs, batch_key=batch_key)



def run_scvi(adata, layer="counts", batch_key="batch", gene_likelihood="nb", n_layers=2, n_latent=30, output_key="scVI", return_model=False, return_corrected=False, transform_batch=None):
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
                              n_layers=n_layers, n_latent=n_latent, output_key="scVI")
    
    # Set up and train the SCANVI model
    lvae = scvi.model.SCANVI.from_scvi_model(scvi_model, adata=adata, labels_key=labels_key, unlabeled_category=unlabeled_category)
    lvae.train(max_epochs=20, n_samples_per_label=100)
    
    if return_corrected:
        corrected_expression = lvae.get_normalized_expression(transform_batch=transform_batch)
        adata.layers[output_key + "_corrected" + ("" if transform_batch is None else f"_{transform_batch}")] = corrected_expression
    # Store the SCANVI latent representation in the specified obsm key
    adata.obsm[output_key] = lvae.get_latent_representation()
    




