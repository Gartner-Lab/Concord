
from .. import logger

def run_umap(adata,
             source_key='encoded', umap_key='encoded_UMAP',
             n_components=2, n_pc=None,
             n_neighbors=30, min_dist=0.1,
             metric='euclidean', spread=1.0, n_epochs=None,
             random_state=0, use_cuml=False):

    if use_cuml:
        try:
            from cuml.manifold import UMAP as cumlUMAP
            umap_model = cumlUMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                                  spread=spread, n_epochs=n_epochs, random_state=random_state)
        except ImportError:
            logger.warning("cuML is not available. Falling back to standard UMAP.")
            umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                                   spread=spread, n_epochs=n_epochs, random_state=random_state)
    else:
        import umap
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                               spread=spread, n_epochs=n_epochs, random_state=random_state)

    if source_key in adata.obsm:
        source_data = adata.obsm[source_key]
    elif source_key in adata.layers:
        source_data = adata.layers[source_key]
    else:
        raise ValueError(f"Source key '{source_key}' not found in adata.obsm or adata.layers")
    
    if n_pc is not None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_pc, random_state=random_state)
        source_data = pca.fit_transform(source_data)
        logger.info(f"PCA performed on source data with {n_pc} components")

    adata.obsm[umap_key] = umap_model.fit_transform(source_data)
    logger.info(f"UMAP embedding stored in adata.obsm['{umap_key}']")




       