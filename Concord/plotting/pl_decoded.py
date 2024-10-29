import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

from .pl_heatmap import heatmap_with_annotations


def plot_adata_layer_heatmaps(adata, ncells=None, ngenes=None, layers=['X_concord_decoded', 'X_log1p'], 
                              transpose=False,
                              obs_keys = None, 
                              cluster_rows=False, cluster_cols=False,
                              use_clustermap=False,
                              seed=0, figsize=(6,6), cmap='viridis', 
                              dpi=300, 
                              save_path=None):
    import seaborn as sns
    import scipy.sparse as sp

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

    # Subsample cells if necessary
    if ncells < adata.shape[0]:
        subsampled_adata = sc.pp.subsample(adata, n_obs=ncells, copy=True)
    else:
        subsampled_adata = adata

    # Subsample genes if necessary
    if ngenes < adata.shape[1]:
        subsampled_genes = np.random.choice(subsampled_adata.var_names, size=ngenes, replace=False)
        subsampled_adata = subsampled_adata[:, subsampled_genes]
    else:
        subsampled_adata = adata

    # Determine the number of columns in the subplots
    ncols = len(layers)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, ncols, figsize=(figsize[0] * ncols, figsize[1]))

    # Plot heatmaps for each layer
    glist = []
    for i, layer in enumerate(layers):
        if layer == 'X':
            x = subsampled_adata.X
        elif layer in subsampled_adata.layers.keys():
            x = subsampled_adata.layers[layer]
        else:
            raise ValueError(f"Layer '{layer}' not found in adata")
        if sp.issparse(x):
            x = x.toarray()

        if use_clustermap:
            g = heatmap_with_annotations(
                subsampled_adata, 
                x, 
                transpose=transpose, 
                obs_keys=obs_keys, 
                cmap=cmap, 
                cluster_rows=cluster_rows, 
                cluster_cols=cluster_cols, 
                value_annot=False, 
                figsize=figsize,
                show=False
            )
            
            # Save the clustermap figure to a buffer
            from io import BytesIO
            buf = BytesIO()
            g.figure.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)

            # Load the image from the buffer and display it in the subplot
            import matplotlib.image as mpimg
            img = mpimg.imread(buf)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Heatmap of {layer}')

            # Close the clustermap figure to free memory
            plt.close(g.figure)
            buf.close()
        else:
            sns.heatmap(x, cmap=cmap, ax=axes[i])
            axes[i].set_title(f'Heatmap of {layer}')

    if save_path:
        plt.savefig(save_path, dpi=dpi)

    plt.show()
