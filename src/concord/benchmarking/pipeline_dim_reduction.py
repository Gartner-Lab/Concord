
from .time_memory import Timer
from . import logger


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
    concord_epochs=15
):
    from ..utils import run_pca, run_umap, run_tsne, run_diffusion_map, run_NMF, run_SparsePCA, \
    run_FactorAnalysis, run_FastICA, run_LDA, run_zifa, run_phate
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
        time_log['Concord'] = safe_run("Concord", Concord(use_decoder=False, **concord_args).fit_transform, output_key='Concord')

    if "Concord-decoder" in methods:
        time_log['Concord-decoder'] = safe_run("Concord-decoder", Concord(use_decoder=True, **concord_args).fit_transform, output_key='Concord-decoder')

    if "Concord-pknn0" in methods:
        time_log['Concord-pknn0'] = safe_run("Concord-pknn0", Concord(use_decoder=False, p_intra_knn=0.0, **concord_args).fit_transform, output_key='Concord-pknn0')
    # Save the time log
    time_log_path = os.path.join(save_dir, "dimensionality_reduction_timelog.json")
    with open(time_log_path, "w") as f:
        import json
        json.dump(time_log, f, indent=4)
    logger.info(f"Time log saved to: {time_log_path}")

    return time_log


