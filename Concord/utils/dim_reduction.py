
from .. import logger

def run_umap(adata,
             source_key='encoded', umap_key='encoded_UMAP',
             n_components=2, n_pc=None,
             n_neighbors=30, min_dist=0.1,
             metric='euclidean', spread=1.0, n_epochs=None,
             random_state=0, use_cuml=False):

    if n_pc is not None:
        run_pca(adata, source_key=source_key, result_key='PCA', n_pc=n_pc)
        source_data = adata.obsm['PCA']
    else:
        source_data = adata.obsm[source_key]

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

    
    adata.obsm[umap_key] = umap_model.fit_transform(source_data)
    logger.info(f"UMAP embedding stored in adata.obsm['{umap_key}']")



def run_pca(adata, source_key='encoded', 
            result_key="PCA", random_state=0,
            n_pc=50, svd_solver='auto'):
    from sklearn.decomposition import PCA

    # Extract the data from obsm
    if source_key in adata.obsm:
        source_data = adata.obsm[source_key]
    elif source_key in adata.layers:
        source_data = adata.layers[source_key]
    elif source_key == 'X':
        source_data = adata.X
    else:
        raise ValueError(f"Source key '{source_key}' not found in adata.obsm or adata.layers")
    
    pca = PCA(n_components=n_pc, random_state=random_state, svd_solver=svd_solver)
    source_data = pca.fit_transform(source_data)
    logger.info(f"PCA performed on source data with {n_pc} components")

    if result_key is None:
        result_key = f"PCA_{n_pc}"
    adata.obsm[result_key] = source_data
    logger.info(f"PCA embedding stored in adata.obsm['{result_key}']")

    return adata
