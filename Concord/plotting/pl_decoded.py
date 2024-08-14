import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp

def plot_adata_layer_heatmaps(adata, ncells=None, ngenes=None, layers=['X_concord_decoded', 'X_log1p'], 
                        seed=0, figsize=(6,6), cmap='viridis', dpi=300, save_path=None):
    # If ncells is None, plot all cells
    if ncells is None:
        ncells = adata.shape[0]
    # If ngenes is None, plot all genes
    if ngenes is None:
        ngenes = adata.shape[1]

    # Check if ncells and ngenes are greater than adata.shape
    if ncells > adata.shape[0]:
        raise ValueError(f"ncells ({ncells}) is greater than the number of cells in adata ({adata.shape[0]})")
    if ngenes > adata.shape[1]:
        raise ValueError(f"ngenes ({ngenes}) is greater than the number of genes in adata ({adata.shape[1]})")

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Subsample cells
    subsampled_adata = sc.pp.subsample(adata, n_obs=ncells, copy=True)

    # Subsample genes
    subsampled_genes = np.random.choice(subsampled_adata.var_names, size=ngenes, replace=False)
    subsampled_adata = subsampled_adata[:, subsampled_genes]

    # Determine the number of columns in the subplots
    ncols = len(layers)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, ncols, figsize=(figsize[0] * ncols, figsize[1]))

    # Plot heatmaps for each layer
    for i, layer in enumerate(layers):
        x = subsampled_adata.layers[layer]
        if sp.issparse(x):
            x = x.toarray()
        sns.heatmap(x, cmap=cmap, ax=axes[i])
        axes[i].set_title(f'Heatmap of {layer}')

    if save_path:
        plt.savefig(save_path, dpi=dpi)

    plt.show()
