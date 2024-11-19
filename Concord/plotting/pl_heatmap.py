from .palettes import get_color_mapping
from matplotlib.patches import Patch

def add_legend(ax, labels, palette, title=None, fontsize=8, bbox_anchor=(1, 1)):
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
    handles = [Patch(facecolor=color, edgecolor='none') for color in palette]
    ax.legend(handles, labels, title=title, loc='upper left', fontsize=fontsize,
              title_fontsize=fontsize, bbox_to_anchor=bbox_anchor, borderaxespad=0)


def heatmap_with_annotations(adata, val, transpose=True, obs_keys=None, 
                             cmap='viridis', vmin=None, vmax=None, 
                             cluster_rows=True, cluster_cols=True, pal=None, add_color_legend=False,
                             value_annot=False, title=None, title_fontsize=16,
                             yticklabels=True, xticklabels=False, 
                             use_clustermap=True, ax=None,
                             figsize=(12, 8), dpi=300, show=True, save_path=None):
    """
    Create a heatmap colored by multiple columns in adata.obs and optionally save the figure.

    Parameters:
    - adata: AnnData object
    - val: Data source for heatmap, can be 'X', a layer name, an obsm key, a DataFrame, or a numpy array
    - obs_keys: List of column names in adata.obs to use for coloring
    - cmap: Colormap for the heatmap
    - figsize: Size of the figure
    - save_path: Path to save the figure. If None, the figure is not saved.
    """
    import seaborn as sns
    import scipy.sparse as sp
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.colors as mcolors

    if not isinstance(pal, dict):
        pal = {col: pal for col in obs_keys}

    # Check if val is string or a matrix
    if isinstance(val, str):
        if val == 'X':
            data = pd.DataFrame(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
        elif val in adata.layers.keys():
            data = pd.DataFrame(adata.layers[val].toarray() if sp.issparse(adata.layers[val]) else adata.layers[val])
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
                use_colors[col] = data_col.map(dict(zip(data_col.unique(), palette))).to_numpy()       
                if add_color_legend:
                    unique_labels = data_col.unique()
                    legend_data.append((unique_labels, palette, col))
                
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
            figsize=figsize if ax is None else None,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            yticklabels=yticklabels,
            xticklabels=xticklabels
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

    if add_color_legend and legend_data:
        for labels, palette, title in legend_data:
            add_legend(ax, labels, palette, title=title, bbox_anchor=(1, 1))

    if save_path:
        plt.savefig(save_path, dpi=dpi)

    if show and ax is None:
        plt.show()

    return g if use_clustermap else ax
