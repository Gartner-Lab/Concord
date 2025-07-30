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



def heatmap_with_annotations(
    adata,
    val,
    *,
    transpose=True,
    obs_keys=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    cluster_rows=True,
    cluster_cols=True,
    pal=None,
    add_color_legend=False,
    value_annot=False,
    title=None,
    title_fontsize=16,
    annot_fontsize=8,
    yticklabels=True,
    xticklabels=False,
    use_clustermap=True,
    cluster_method="ward",
    cluster_metric="euclidean",
    rasterize=True,
    ax=None,
    figsize=(12, 8),
    seed=42,
    dpi=300,
    show=True,
    save_path=None,
    # ───────── NEW OPTIONS ─────────
    log_transform=False,
    pseudocount=1e-6,
    row_scale=False,
    col_scale=False,
    clip_limits=None,       # e.g. (-3, 3)  or  (None, 3)  or  None
):
    """
    Plot a heat‑map (optionally via seaborn.clustermap) with colour bars for
    `adata.obs` columns, plus flexible transformation / scaling.

    NEW ARGS
    --------
    log_transform : bool
        If True, apply log10(x + pseudocount) **before** any scaling.
    pseudocount   : float
        Small value added prior to log‑transform (ignored if log_transform=False).
    row_scale     : bool
        Z‑score across each row (genes) *after* log/clip. Mutually exclusive with
        `col_scale`.  Equivalent to `t(scale(t(..)))` in your R helper.
    col_scale     : bool
        Z‑score across each column (cells).  If both row_scale and col_scale are
        True a ValueError is raised.
    clip_limits   : tuple | None
        `(low, high)` limits applied via `np.clip` **after** scaling.  Use
        `None` for one side to leave it un‑clipped, e.g. `(None, 3)`.

    All original parameters remain unchanged.
    """
    # --------------------------------------------------------------
    import seaborn as sns
    import scipy.sparse as sp
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.collections as mcoll
    from sklearn.preprocessing import StandardScaler

    np.random.seed(seed)

    # ----------------------- prepare data -------------------------
    if isinstance(val, str):
        if val == "X":
            data = adata.X
        elif val in adata.layers:
            data = adata.layers[val]
        elif val in adata.obsm:
            data = adata.obsm[val]
        else:
            raise ValueError(f"val '{val}' not found in adata")
        data = data.toarray() if sp.issparse(data) else np.asarray(data)
        data = pd.DataFrame(data)
        # attach var names if gene matrix
        if val in ("X", *adata.layers.keys()) and adata.var_names is not None:
            data.columns = adata.var_names
    elif isinstance(val, pd.DataFrame):
        data = val.copy()
    elif isinstance(val, np.ndarray):
        data = pd.DataFrame(val)
    else:
        raise TypeError("val must be a string, numpy.ndarray, or pandas.DataFrame")

    if transpose:
        data = data.T  # genes become rows
        data.columns = adata.obs_names

    # ------------------- optional transformations -----------------
    if log_transform:
        data = np.log10(data + pseudocount)

    if row_scale and col_scale:
        raise ValueError("Choose row_scale or col_scale, not both")

    # --- row‑wise scaling: final fix ------------------------------------
    if row_scale:
        data = pd.DataFrame(
            StandardScaler(with_mean=True, with_std=True)
            .fit_transform(data.T)   # cells as rows, genes as columns
            .T,                      # back to genes × cells
            index=data.index,
            columns=data.columns,
        )

    if col_scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        data = pd.DataFrame(
            scaler.fit_transform(data.T).T, index=data.index, columns=data.columns
        )

    lo, hi = (None, None)
    if clip_limits is not None:
        lo, hi = clip_limits
        data = data.clip(lower=lo, upper=hi)

    heat_vmin = vmin
    heat_vmax = vmax

    # --------------------- colour‑bar assembly --------------------
    if obs_keys is not None:
        if pal is None:
            pal = {}
        # colours per column
        col_colours = {}
        legends = []
        for key in obs_keys:
            data_col, obs_cmap, palette = get_color_mapping(adata, key, pal, seed=seed)
            # 1️⃣  continuous variable  ------------------------------------------
            if obs_cmap is not None:                              # numeric column
                norm   = mcolors.Normalize(vmin=data_col.min(), vmax=data_col.max())
                colour_list = [mcolors.to_hex(obs_cmap(norm(v))) for v in data_col]
                lut_for_legend = None
            else:                                             # palette is a dict
                colour_list    = data_col.map(palette).tolist()
                lut_for_legend = palette                     # keep for legend

            col_colours[key] = colour_list

            if add_color_legend and lut_for_legend is not None:
                legends.append((key, lut_for_legend))
        col_colors_df = (
            pd.DataFrame(col_colours, index=data.columns)
            if col_colours else None
        )
    else:
        col_colors_df = None
        legends = []

    # ---------------------- plotting section ----------------------
    if use_clustermap:
        g = sns.clustermap(
            data,
            cmap=cmap,
            vmin=heat_vmin,
            vmax=heat_vmax,
            col_colors=col_colors_df if transpose else None,
            row_colors=col_colors_df if not transpose else None,
            annot=value_annot,
            annot_kws={"size": annot_fontsize},
            figsize=figsize if ax is None else None,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            yticklabels=yticklabels,
            xticklabels=xticklabels,
            method=cluster_method,
            metric=cluster_metric,
            rasterized=rasterize,
        )
        ax = g.ax_heatmap
        # rasterize *only* heat‑map cells, keep text/vector on top
        if rasterize:
            for artist in ax.findobj(mcoll.QuadMesh):
                artist.set_rasterized(True)
        if title:
            g.figure.suptitle(title, fontsize=title_fontsize)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data,
            cmap=cmap,
            vmin=heat_vmin,
            vmax=heat_vmax,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            annot=value_annot,
            annot_kws={"size": annot_fontsize},
            ax=ax,
            rasterized=rasterize,
        )
        if title:
            ax.set_title(title, fontsize=title_fontsize)
            
        # ensure only the QuadMesh is rasterised
        if rasterize:
            for artist in ax.findobj(mcoll.QuadMesh):
                artist.set_rasterized(True)

    # ---------------------- add legends ---------------------------
    if add_color_legend and legends:
        for key, lut in legends:
            handles = [Patch(facecolor=c, label=k) for k, c in lut.items()]
            ax.legend(
                handles=handles,
                title=key,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
                fontsize=annot_fontsize,
            )

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

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
