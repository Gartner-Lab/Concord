
import time
import torch
import pandas as pd
from pathlib import Path
from .preprocessor import Preprocessor
from .plotting import Plotter
from ..model.evaluator import eval_scib_metrics
from ..model.clarity import Clarity
import anndata as ad
from .file_io import save_obsm_to_hdf5

def run_clarity_encoder(
    params,
    proj_name,
    data_path,
    save_dir,
    domain_key,
    class_key,
    show_cols,
    umap_n_neighbors=10,
    subset_hvg=20000,
    epochs=5,
    eval_scib=False,
    save_obsm=True
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    preprocessor = Preprocessor(
        use_key="X",
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=False,
        result_log1p_key="X_log1p",
        domain_key=domain_key,
        subset_hvg=subset_hvg
    )

    plotter = Plotter()

    clarity = Clarity(
        proj_name, save_dir=save_dir,
        domain_key=domain_key, class_key=class_key,
        importance_penalty_type='L1',
        eval_epoch_interval=0,
        epochs=epochs, device=device, **params
    )

    adata = ad.read_h5ad(data_path)
    preprocessor(adata)

    # Train and evaluate
    adata = clarity.encode_adata(adata, input_layer_key='X_log1p')

    if eval_scib:
        eval_results = eval_scib_metrics(adata, batch_key=domain_key, label_key=class_key)
    else:
        eval_results = {}

    # UMAP Embeddings
    adata = clarity.run_umap(adata, use_cuml=False, umap_key='encoded_UMAP', n_components=2, n_epochs=300, n_neighbors=umap_n_neighbors, min_dist=0.1, metric='euclidean')

    show_emb = 'encoded_UMAP'
    file_suffix = f"{params['augmentation_mask_prob']}_{params['use_importance_mask']}_{params['importance_penalty_weight']}_{params['batch_size']}_{time.strftime('%b%d-%H%M')}"
    plotter.plot_embeddings(
        adata, show_emb, show_cols, figsize=(17, 11), dpi=600, ncols=3, font_size=3, point_size=10, legend_loc='on data',
        save_path=save_dir / f"embeddings_{show_emb}_{file_suffix}.png"
    )

    plotter.visualize_importance_weights(
        clarity.model, adata, top_n=30, mode='histogram',
        save_path=save_dir / f"importance_weights_{file_suffix}.png"
    )

    adata = clarity.run_umap(adata, use_cuml=False, umap_key='encoded_UMAP_3D', n_components=3, n_epochs=300, n_neighbors=umap_n_neighbors, min_dist=0.1, metric='euclidean')

    show_emb = 'encoded_UMAP_3D'
    for col in show_cols:
        plotter.visualize_3d_embedding(
            adata, embedding_key=show_emb, color_by=col,
            save_path=save_dir / f"embeddings_{show_emb}_{col}_{file_suffix}.html",
            point_size=1, opacity=0.8, width=1500, height=800
        )

    if save_obsm:
        obsm_filename = save_dir / f"obsm_{file_suffix}.h5"
        save_obsm_to_hdf5(adata, obsm_filename)

    config_dict = clarity.config.to_dict()  # Convert Config to dictionary

    return config_dict, eval_results



def log_results(config_dicts, eval_results, save_dir, file_name):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure save_dir is created if it does not exist

    config_df = pd.DataFrame(config_dicts)
    eval_df = pd.DataFrame(eval_results)

    with pd.ExcelWriter(save_dir / file_name) as writer:
        config_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
        eval_df.to_excel(writer, sheet_name='Results', index=False)

    return config_df, eval_df