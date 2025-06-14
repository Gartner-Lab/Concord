
from .. import logger
from . import Timer
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


def safe_run(method_name, func, **kwargs):
    """Wrapper to safely run a method, time it, and log errors if it fails."""
    import traceback
    timer = Timer()
    try:
        with timer:
            func(**kwargs)
        logger.info(f"{method_name}: Successfully completed in {timer.interval:.2f} seconds.")
        used_time = timer.interval
    except Exception as e:
        logger.warning(f"{method_name}: Failed to run. Error: {str(e)}")
        print(traceback.format_exc())
        used_time = None
    
    return used_time


def run_dimensionality_reduction_pipeline(
    adata,
    source_key="X",
    methods=["PCA", "UMAP", "t-SNE", "DiffusionMap", "NMF", "SparsePCA", 
             "FactorAnalysis", "FastICA", "LDA", "ZIFA", "scVI", "PHATE", 
             "Concord", "Concord-decoder", "Concord-pknn0"],
    n_components=10,
    random_state=42,
    device="cpu",
    save_dir="./",
    concord_epochs=15,
    concord_min_pid=0.95
):
    """
    Runs multiple dimensionality reduction techniques on an AnnData object.
    Logs execution time and errors for each method, and saves time log to save_dir.

    Parameters:
        adata: AnnData
            Input AnnData object.
        methods: list
            List of methods to run.
        source_key: str
            The layer or key in adata to use as the source data for methods.
        n_components: int
            Number of components to compute for applicable methods.
        random_state: int
            Random seed for reproducibility.
        device: str
            Device for Concord/scVI computations, e.g., "cpu" or "cuda".
        save_dir: str
            Directory to save the time log and Concord model checkpoints.
        concord_epochs: int
            Number of epochs to train Concord.
        concord_min_pid: float
            Minimum intra-domain probability for Concord.

    Returns:
        dict: Dictionary of output keys for each method and their execution time.
    """
    import os
    from . import run_scvi
    from ..concord import Concord

    os.makedirs(save_dir, exist_ok=True)  # Ensure save_dir exists
    time_log = {}
    seed = random_state
    
    # Core methods
    if "PCA" in methods:
        time_log['PCA'] = safe_run("PCA", run_pca, adata=adata, source_key=source_key, result_key='PCA', n_pc=n_components, random_state=seed)

    if "UMAP" in methods:
        time_log['UMAP'] = safe_run("UMAP", run_umap, adata=adata, source_key=source_key, result_key='UMAP', random_state=seed)

    if "t-SNE" in methods:
        time_log['t-SNE'] = safe_run("t-SNE", run_tsne, adata=adata, source_key=source_key, result_key='tSNE', random_state=seed)

    if "DiffusionMap" in methods:
        time_log['DiffusionMap'] = safe_run("DiffusionMap", run_diffusion_map, adata=adata, source_key=source_key, n_neighbors=15, n_components=n_components, result_key='DiffusionMap', seed=seed)

    if "NMF" in methods:
        time_log['NMF'] = safe_run("NMF", run_NMF, adata=adata, source_key=source_key, n_components=n_components, result_key='NMF', seed=seed)

    if "SparsePCA" in methods:
        time_log['SparsePCA'] = safe_run("SparsePCA", run_SparsePCA, adata=adata, source_key=source_key, n_components=n_components, result_key='SparsePCA', seed=seed)

    if "FactorAnalysis" in methods:
        time_log['FactorAnalysis'] = safe_run("FactorAnalysis", run_FactorAnalysis, adata=adata, source_key=source_key, n_components=n_components, result_key='FactorAnalysis', seed=seed)

    if "FastICA" in methods:
        time_log['FastICA'] = safe_run("FastICA", run_FastICA, adata=adata, source_key=source_key, result_key='FastICA', n_components=n_components, seed=seed)

    if "LDA" in methods:
        time_log['LDA'] = safe_run("LDA", run_LDA, adata=adata, source_key=source_key, result_key='LDA', n_components=n_components, seed=seed)

    if "ZIFA" in methods:
        time_log['ZIFA'] = safe_run("ZIFA", run_zifa, adata=adata, source_key=source_key, log=True, result_key='ZIFA', n_components=n_components)

    if "scVI" in methods:
        time_log['scVI'] = safe_run("scVI", run_scvi, adata=adata, batch_key=None, output_key='scVI', return_model=False, return_corrected=False, transform_batch=None)

    if "PHATE" in methods:
        time_log['PHATE'] = safe_run("PHATE", run_phate, adata=adata, layer=source_key, n_components=2, result_key='PHATE', seed=seed)

    # Concord methods
    concord_args = {
        'adata': adata,
        'input_feature': None,
        'latent_dim': n_components,
        'n_epochs': concord_epochs,
        'domain_key': None,
        'seed': seed,
        'device': device,
        'save_dir': save_dir
    }
    if "Concord" in methods:
        time_log['Concord'] = safe_run("Concord", Concord(use_decoder=False, **concord_args).encode_adata, input_layer_key=source_key, output_key='Concord', preprocess=False)

    if "Concord-decoder" in methods:
        time_log['Concord-decoder'] = safe_run("Concord-decoder", Concord(use_decoder=True, **concord_args).encode_adata, input_layer_key=source_key, output_key='Concord-decoder', preprocess=False)

    if "Concord-pknn0" in methods:
        time_log['Concord-pknn0'] = safe_run("Concord-pknn0", Concord(use_decoder=False, p_intra_knn=0.0, **concord_args).encode_adata, input_layer_key=source_key, output_key='Concord-pknn0', preprocess=True)
    # Save the time log
    time_log_path = os.path.join(save_dir, "dimensionality_reduction_timelog.json")
    with open(time_log_path, "w") as f:
        import json
        json.dump(time_log, f, indent=4)
    logger.info(f"Time log saved to: {time_log_path}")

    return time_log
