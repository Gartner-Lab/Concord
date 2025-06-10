from .palettes import get_color_mapping
from matplotlib.patches import Patch
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt



def add_legend(ax, palette, title=None, fontsize=8, bbox_anchor=(1, 1)):
    """
    Adds a color legend directly to the plot for categorical data.
    
    Parameters:
    - ax: The matplotlib axis where the legend will be added.
    - labels: List of category labels.
    - palette: List of colors corresponding to the labels.
    - title: Title for the legend.
    - fontsize: Font size for the legend.
    - bbox_anchor: Location of the legend box.
    """
    handles = [Patch(facecolor=color, edgecolor='none') for color in palette.values()]
    labels = [key for key in palette]
    ax.legend(handles, labels, title=title, loc='upper left', fontsize=fontsize,
              title_fontsize=fontsize, bbox_to_anchor=bbox_anchor, borderaxespad=0)


def heatmap_with_annotations(adata, val, transpose=True, obs_keys=None, 
                             cmap='viridis', vmin=None, vmax=None, 
                             cluster_rows=True, cluster_cols=True, pal=None, add_color_legend=False,
                             value_annot=False, title=None, title_fontsize=16, annot_fontsize=8,
                             yticklabels=True, xticklabels=False, 
                             use_clustermap=True, 
                             cluster_method='ward',        
                             cluster_metric='euclidean',
                             rasterize=True,
                             ax=None,
                             figsize=(12, 8), seed=42,
                             dpi=300, show=True, save_path=None):
    """
    Creates a heatmap colored by multiple columns in `adata.obs`, optionally clusters the rows/columns, 
    and provides categorical or continuous annotations.

    Args:
        adata (AnnData): 
            AnnData object containing the dataset.
        val (str | np.ndarray | pd.DataFrame): 
            Data source for heatmap. Can be:
                - `'X'`: Uses `adata.X`
                - A layer name from `adata.layers`
                - An embedding from `adata.obsm`
                - A `numpy.ndarray` or `pandas.DataFrame`
        transpose (bool, optional): 
            If `True`, transposes the data matrix. Defaults to `True`.
        obs_keys (list, optional): 
            List of column names in `adata.obs` to use for categorical or numerical coloring. Defaults to `None`.
        cmap (str, optional): 
            Colormap for heatmap values. Defaults to `'viridis'`.
        vmin (float, optional): 
            Minimum value for color scaling. Defaults to `None`.
        vmax (float, optional): 
            Maximum value for color scaling. Defaults to `None`.
        cluster_rows (bool, optional): 
            Whether to cluster rows. Defaults to `True`.
        cluster_cols (bool, optional): 
            Whether to cluster columns. Defaults to `True`.
        pal (dict, optional): 
            Dictionary mapping category values to colors. Defaults to `None`.
        add_color_legend (bool, optional): 
            If `True`, adds a legend for categorical annotations. Defaults to `False`.
        value_annot (bool, optional): 
            If `True`, annotates each heatmap cell with values. Defaults to `False`.
        title (str, optional): 
            Title of the heatmap. Defaults to `None`.
        title_fontsize (int, optional): 
            Font size for title. Defaults to `16`.
        annot_fontsize (int, optional): 
            Font size for annotations (if `value_annot=True`). Defaults to `8`.
        yticklabels (bool, optional): 
            Whether to show row labels. Defaults to `True`.
        xticklabels (bool, optional): 
            Whether to show column labels. Defaults to `False`.
        use_clustermap (bool, optional): 
            If `True`, uses `seaborn.clustermap` for hierarchical clustering. Otherwise, uses `sns.heatmap`. Defaults to `True`.
        cluster_method (str, optional): 
            Clustering method for hierarchical clustering (e.g., `'ward'`, `'average'`, `'single'`). Defaults to `'ward'`.
        cluster_metric (str, optional): 
            Distance metric for hierarchical clustering (e.g., `'euclidean'`, `'correlation'`). Defaults to `'euclidean'`.
        rasterize (bool, optional): 
            If `True`, rasterizes heatmap elements for efficient plotting. Defaults to `True`.
        ax (matplotlib.axes.Axes, optional): 
            Matplotlib Axes object to plot on. Defaults to `None`.
        figsize (tuple, optional): 
            Size of the figure `(width, height)`. Defaults to `(12, 8)`.
        seed (int, optional): 
            Random seed for reproducibility. Defaults to `42`.
        dpi (int, optional): 
            Resolution of the saved figure. Defaults to `300`.
        show (bool, optional): 
            If `True`, displays the plot. Defaults to `True`.
        save_path (str, optional): 
            Path to save the figure. If `None`, the figure is not saved. Defaults to `None`.

    Returns:
        matplotlib.Axes | seaborn.ClusterGrid: 
            - If `use_clustermap=True`, returns a `seaborn.ClusterGrid` object.
            - Otherwise, returns a `matplotlib.Axes` object.
    """

    import seaborn as sns
    import scipy.sparse as sp
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.colors as mcolors
    import matplotlib.collections as mcoll

    np.random.seed(seed)
    if not isinstance(pal, dict):
        pal = {col: pal for col in obs_keys}

    # Check if val is string or a matrix
    if isinstance(val, str):
        if val == 'X':
            data = pd.DataFrame(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
            # Set the index to the gene names if available
            if adata.var_names is not None:
                data.columns = adata.var_names
        elif val in adata.layers.keys():
            data = pd.DataFrame(adata.layers[val].toarray() if sp.issparse(adata.layers[val]) else adata.layers[val])
            # Set the index to the gene names if available
            if adata.var_names is not None:
                data.columns = adata.var_names
        elif val in adata.obsm.keys():
            data = pd.DataFrame(adata.obsm[val])
        else:
            raise ValueError(f"val '{val}' not found in adata")
    elif isinstance(val, pd.DataFrame):
        data = val.reset_index(drop=True)
    elif isinstance(val, np.ndarray):
        data = pd.DataFrame(val)
    else:
        raise ValueError("val must be a string, pandas DataFrame, or numpy array")
    
    if transpose:
        data = data.T
    
    if obs_keys is not None:
        colors_df = adata.obs[obs_keys].copy()
        use_colors = pd.DataFrame(index=colors_df.index)
        legend_data = []
        for col in obs_keys:
            data_col = colors_df[col]
            data_col, col_cmap, palette = get_color_mapping(adata, col, pal)
            if pd.api.types.is_numeric_dtype(data_col):
                norm = mcolors.Normalize(vmin=data_col.min(), vmax=data_col.max())
                use_colors[col] = [col_cmap(norm(val)) for val in data_col]
            else:
                use_colors[col] = data_col.map(palette).to_numpy()       
                if add_color_legend:
                    legend_data.append((palette, col))

        use_colors.reset_index(drop=True, inplace=True)
    else:
        use_colors = None

    if ax is None and not use_clustermap:
        fig, ax = plt.subplots(figsize=figsize)

    if use_clustermap:
        g = sns.clustermap(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            col_colors=use_colors if transpose else None,
            row_colors=use_colors if not transpose else None,
            annot=value_annot,
            annot_kws={"size": annot_fontsize},
            figsize=figsize if ax is None else None,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            yticklabels=yticklabels,
            xticklabels=xticklabels,
            method=cluster_method,
            metric=cluster_metric,
        )
        ax = g.ax_heatmap
        if title:
            g.figure.suptitle(title, fontsize=title_fontsize)
    else:
        sns.heatmap(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            #cbar_kws={'label': 'Expression'},
            annot=value_annot,
            fmt='.2f',
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            ax=ax
        )
        cbar = ax.collections[0].colorbar  # Access the color bar
        cbar.ax.tick_params(labelsize=title_fontsize-2)
        if title:
            ax.set_title(title, fontsize=title_fontsize)

    # Rasterize only the heatmap cells
    if rasterize:
        for artist in ax.get_children():
            if isinstance(artist, mcoll.QuadMesh):
                artist.set_rasterized(True)

    if add_color_legend and legend_data:
        for palette, title in legend_data:
            add_legend(ax, palette, title=title, bbox_anchor=(1, 1))

    if save_path:
        plt.savefig(save_path, dpi=dpi)

    if show and ax is None:
        plt.show()

    return g if use_clustermap else ax





def plot_adata_layer_heatmaps(adata, ncells=None, ngenes=None, layers=['X_concord_decoded', 'X_log1p'], 
                              transpose=False,
                              obs_keys = None, 
                              cluster_rows=False, cluster_cols=False,
                              use_clustermap=False,
                              seed=0, figsize=(6,6), cmap='viridis', 
                              dpi=300, vmin=None, vmax=None,
                              save_path=None):
    """
    Plots heatmaps of selected layers from an AnnData object, optionally clustering rows and columns.

    This function visualizes gene expression data from different layers of an AnnData object as heatmaps.
    It allows for subsampling of cells and genes, clustering of rows and columns, and saving the output
    figure.

    Args:
        adata (AnnData): 
            The AnnData object containing gene expression data.
        ncells (int, optional): 
            Number of cells to subsample. If None, uses all cells. Defaults to `None`.
        ngenes (int, optional): 
            Number of genes to subsample. If None, uses all genes. Defaults to `None`.
        layers (list of str, optional): 
            List of layer names to plot heatmaps for. Defaults to `['X_concord_decoded', 'X_log1p']`.
        transpose (bool, optional): 
            If True, transposes the heatmap (genes as columns). Defaults to `False`.
        obs_keys (list of str, optional): 
            List of categorical metadata columns from `adata.obs` to annotate along heatmap axes. Defaults to `None`.
        cluster_rows (bool, optional): 
            Whether to cluster rows (genes). Defaults to `False`.
        cluster_cols (bool, optional): 
            Whether to cluster columns (cells). Defaults to `False`.
        use_clustermap (bool, optional): 
            If True, uses `seaborn.clustermap` instead of `sns.heatmap` for hierarchical clustering. Defaults to `False`.
        seed (int, optional): 
            Random seed for reproducibility in subsampling. Defaults to `0`.
        figsize (tuple, optional): 
            Figure size `(width, height)`. Defaults to `(6, 6)`.
        cmap (str, optional): 
            Colormap for the heatmap. Defaults to `'viridis'`.
        dpi (int, optional): 
            Resolution of the saved figure. Defaults to `300`.
        vmin (float, optional): 
            Minimum value for heatmap normalization. Defaults to `None`.
        vmax (float, optional): 
            Maximum value for heatmap normalization. Defaults to `None`.
        save_path (str, optional): 
            If provided, saves the heatmap figure to the specified path. Defaults to `None`.

    Raises:
        ValueError: If `ncells` or `ngenes` is greater than the dimensions of `adata`.
        ValueError: If a specified `layer` is not found in `adata.layers`.

    Returns:
        None
            Displays the heatmaps and optionally saves the figure.

    Example:
        ```python
        plot_adata_layer_heatmaps(adata, ncells=500, ngenes=100, layers=['X', 'X_log1p'],
                                  cluster_rows=True, cluster_cols=True, use_clustermap=True,
                                  save_path="heatmap.png")
        ```
    """

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
                vmin=vmin,
                vmax=vmax,
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
            sns.heatmap(x, 
                        cmap=cmap, 
                        vmin=vmin, 
                        vmax=vmax,
                        ax=axes[i])
            axes[i].set_title(f'Heatmap of {layer}')

    if save_path:
        plt.savefig(save_path, dpi=dpi)

    plt.show()
