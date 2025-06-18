
from .. import logger
from . import get_adata_basis

def run_umap(adata,
             source_key='encoded', result_key='encoded_UMAP',
             n_components=2, n_pc=None,
             n_neighbors=30, min_dist=0.1,
             metric='euclidean', spread=1.0, n_epochs=None,
             random_state=0, use_cuml=False):

    if n_pc is not None:
        run_pca(adata, source_key=source_key, result_key=f'{source_key}_PCA', n_pc=n_pc)
        source_data = adata.obsm[f'{source_key}_PCA']
    else:
        source_data = get_adata_basis(adata, basis=source_key)

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

    
    adata.obsm[result_key] = umap_model.fit_transform(source_data)
    logger.info(f"UMAP embedding stored in adata.obsm['{result_key}']")



def run_pca(adata, source_key='encoded', 
            result_key="PCA", random_state=0,
            n_pc=50, svd_solver='auto'):
    from sklearn.decomposition import PCA

    # Extract the data from obsm
    source_data = get_adata_basis(adata, basis=source_key)
    
    pca = PCA(n_components=n_pc, random_state=random_state, svd_solver=svd_solver)
    pca_res = pca.fit_transform(source_data)
    logger.info(f"PCA performed on source data with {n_pc} components")

    if result_key is None:
        result_key = f"PCA_{n_pc}"
    adata.obsm[result_key] = pca_res
    logger.info(f"PCA embedding stored in adata.obsm['{result_key}']")

    return adata


def run_tsne(adata,
             source_key='encoded', result_key='encoded_tSNE',
             n_components=2, n_pc=None,
             metric='euclidean', perplexity=30, early_exaggeration=12,
             random_state=0):

    if n_pc is not None:
        run_pca(adata, source_key=source_key, result_key=f'{source_key}_PCA', n_pc=n_pc)
        source_data = adata.obsm[f'{source_key}_PCA']
    else:
        source_data = get_adata_basis(adata, basis=source_key)

    import sklearn.manifold
    tsne = sklearn.manifold.TSNE(n_components=n_components, metric=metric, perplexity=perplexity,
                                 early_exaggeration=early_exaggeration, random_state=random_state)

    
    adata.obsm[result_key] = tsne.fit_transform(source_data)
    logger.info(f"T-SNE embedding stored in adata.obsm['{result_key}']")


def run_diffusion_map(adata, source_key='X', n_components=10, n_neighbors=15, result_key='DiffusionMap', seed=42):
    import scanpy as sc
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=source_key)  # Build graph
    sc.tl.diffmap(adata, n_comps=n_components, random_state=seed)
    adata.obsm[result_key] = adata.obsm['X_diffmap']



def run_NMF(adata, source_key='X', n_components=10, result_key='NMF', seed=42):
    from sklearn.decomposition import NMF
    source_data = get_adata_basis(adata, basis=source_key)
    nmf = NMF(n_components=n_components, random_state=seed)
    nmf_res = nmf.fit_transform(source_data)
    adata.obsm[result_key] = nmf_res


def run_SparsePCA(adata, source_key='X', n_components=10, result_key='SparsePCA', seed=42):
    from sklearn.decomposition import SparsePCA
    source_data = get_adata_basis(adata, basis=source_key)
    spca = SparsePCA(n_components=n_components, random_state=seed)
    spca_res = spca.fit_transform(source_data)
    adata.obsm[result_key] = spca_res


def run_FactorAnalysis(adata, source_key='X', n_components=10, result_key='FactorAnalysis', seed=42):
    from sklearn.decomposition import FactorAnalysis
    source_data = get_adata_basis(adata, basis=source_key)
    fa = FactorAnalysis(n_components=n_components, random_state=seed)
    fa_res = fa.fit_transform(source_data)
    adata.obsm[result_key] = fa_res


def run_phate(adata, layer="counts", n_components=2, result_key = 'PHATE', seed=42):
    import phate
    import scprep
    phate_operator = phate.PHATE(n_components=n_components, random_state=seed)
    count_mtx = adata.layers[layer] if layer in adata.layers else adata.X
    # Use recommended sqrt for phate
    count_sqrt = scprep.transform.sqrt(count_mtx)
    adata.obsm[result_key] = phate_operator.fit_transform(count_sqrt)


def run_zifa(adata, n_components=10, source_key='X', log=True, result_key='ZIFA', block_zifa=False):
    from ZIFA import ZIFA, block_ZIFA
    import numpy as np
    # Ensure dense data
    Y = get_adata_basis(adata, basis=source_key)
    if log:
        Y = np.log2(Y + 1)
    # Run ZIFA
    if block_zifa:
        Z, _ = block_ZIFA.fitModel(Y, n_components)
    else:
        Z, _ = ZIFA.fitModel(Y, n_components)
    
    adata.obsm[result_key] = Z
    

def run_FastICA(adata, n_components=10, source_key='X', result_key='FastICA', seed=42):
    from sklearn.decomposition import FastICA
    source_data = get_adata_basis(adata, basis=source_key)
    ica = FastICA(n_components=n_components, random_state=seed)
    ica_res = ica.fit_transform(source_data)
    adata.obsm[result_key] = ica_res


def run_LDA(adata, n_components=10, source_key='X', result_key='LDA', seed=42):
    from sklearn.decomposition import LatentDirichletAllocation
    source_data = get_adata_basis(adata, basis=source_key)
    lda = LatentDirichletAllocation(n_components=n_components, random_state=seed)
    lda_res = lda.fit_transform(source_data)
    adata.obsm[result_key] = lda_res

