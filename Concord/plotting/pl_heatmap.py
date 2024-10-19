

def heatmap_with_annotations(adata, val, obs_keys = None, cmap='viridis', cluster_rows=True, cluster_cols=True, figsize=(12, 8), dpi=300, save_path=None):
    """
    Create a heatmap colored by multiple columns in adata.obs and optionally save the figure.

    Parameters:
    - adata: AnnData object
    - obsm_key: Key in adata.obsm to use for clustering
    - obs_keys: List of column names in adata.obs to use for coloring
    - cmap: Colormap for the heatmap
    - figsize: Size of the figure
    - save_path: Path to save the figure. If None, the figure is not saved.
    """
    # Extract the data for heatmap
    import seaborn as sns
    import scipy.sparse as sp
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import pandas as pd

    # Check if val is string or a matrix
    if isinstance(val, str):
        if val == 'X':
            if sp.issparse(adata.X):
                data = pd.DataFrame(adata.X.toarray().T)
            else:
                data = pd.DataFrame(adata.X.T)
        elif val in adata.layers.keys():
            if sp.issparse(adata.layers[val]):
                data = pd.DataFrame(adata.layers[val].toarray().T)
            else:
                data = pd.DataFrame(adata.layers[val].T)
        elif val in adata.obsm.keys():
            data = pd.DataFrame(adata.obsm[val].T)
        else:
            raise ValueError(f"Key '{key}' not found in adata")
    elif isinstance(val, pd.DataFrame):
        data = val.T
        data.reset_index(drop=True, inplace=True)
    elif  isinstance(val, np.ndarray):
        data = pd.DataFrame(val.T)
    else:
        raise ValueError("val must be a string, pandas DataFrame, or numpy array")
    
    assert data.shape[1] == adata.n_obs, "Data matrix must have the same number of rows as adata"
    
    if obs_keys is not None:
        # Create a dataframe for row colors
        col_colors_df = adata.obs[obs_keys].copy()

        # Create color palettes for each column in obs_keys
        col_colors = pd.DataFrame(index=col_colors_df.index)
        for col in obs_keys:
            data_col = col_colors_df[col]

            if pd.api.types.is_numeric_dtype(data_col):
                # Normalize the numeric data
                norm = Normalize(vmin=data_col.min(), vmax=data_col.max())
                cmap_numeric = plt.get_cmap('RdBu_r')
                col_colors[col] = [cmap_numeric(norm(val)) for val in data_col]

            else:
                data_col = data_col.astype(str)
                unique_values = data_col.dropna().unique()
                palette = sns.color_palette("husl", len(unique_values))
                lut = dict(zip(unique_values, palette))
                # Handle missing data ('NA')  if any
                if data_col.isnull().sum() > 0:
                    lut['NA'] = (1, 1, 1)
                    data_col = data_col.fillna('NA')
                
                col_colors[col] = data_col.map(lut).to_numpy()
                
        col_colors.reset_index(drop=True, inplace=True)
    else:
        col_colors = None

    # Create the heatmap with col_colors
    g = sns.clustermap(
        data,
        cmap=cmap,
        col_colors=col_colors,
        annot=False,
        figsize=figsize,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols
    )

    if save_path:
        plt.savefig(save_path, dpi=dpi)

    plt.show()

