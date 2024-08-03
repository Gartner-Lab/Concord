import umap
from .. import logger

def run_umap(adata,
             source_key='encoded', umap_key='encoded_UMAP',
             n_components=2, n_neighbors=30, min_dist=0.1,
             metric='euclidean', spread=1.0, n_epochs=500,
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
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                               spread=spread, n_epochs=n_epochs, random_state=random_state)

    adata.obsm[umap_key] = umap_model.fit_transform(adata.obsm[source_key])
    logger.info(f"UMAP embedding stored in self.adata.obsm['{umap_key}']")
