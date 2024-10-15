
import matplotlib.pyplot as plt
import pandas as pd


def clustermap_with_annotations(adata, key, obs_keys, cmap='viridis', figsize=(12, 8), dpi=300, save_path=None):
    """
    Create a clustermap colored by multiple columns in adata.obs and optionally save the figure.

    Parameters:
    - adata: AnnData object
    - obsm_key: Key in adata.obsm to use for clustering
    - obs_keys: List of column names in adata.obs to use for coloring
    - cmap: Colormap for the heatmap
    - figsize: Size of the figure
    - save_path: Path to save the figure. If None, the figure is not saved.
    """
    # Extract the data for clustermap
    import seaborn as sns
    import scipy.sparse as sp
    if key == 'X':
        if sp.issparse(adata.X):
            data = pd.DataFrame(adata.X.toarray().T)
        else:
            data = pd.DataFrame(adata.X.T)
    elif key in adata.layers.keys():
        if sp.issparse(adata.layers[key]):
            data = pd.DataFrame(adata.layers[key].toarray().T)
        else:
            data = pd.DataFrame(adata.layers[key].T)
    elif key in adata.obsm.keys():
        data = pd.DataFrame(adata.obsm[key].T)
    else:
        raise ValueError(f"Key '{key}' not found in adata")
    
    # Create a dataframe for row colors
    col_colors_df = adata.obs[obs_keys].copy()

    # Create color palettes for each column in obs_keys
    col_colors = pd.DataFrame(index=col_colors_df.index)
    for col in obs_keys:
        unique_values = col_colors_df[col].dropna().unique()
        palette = sns.color_palette("husl", len(unique_values))
        lut = dict(zip(unique_values, palette))

        # Handle categorical data properly
        if isinstance(col_colors_df[col].dtype, pd.CategoricalDtype):
            col_colors_df[col] = col_colors_df[col].cat.add_categories('NA')

        lut['NA'] = (1, 1, 1)
        col_colors_df[col] = col_colors_df[col].fillna('NA')
        col_colors[col] = col_colors_df[col].astype(str).map(lut).to_numpy()

    col_colors.reset_index(drop=True, inplace=True)

    # Create the clustermap with col_colors
    g = sns.clustermap(
        data,
        cmap=cmap,
        col_colors=col_colors,
        annot=False,
        figsize=figsize
    )

    if save_path:
        plt.savefig(save_path, dpi=dpi)

    plt.show()

