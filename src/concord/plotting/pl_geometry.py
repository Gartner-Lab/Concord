import matplotlib.pyplot as plt
import numpy as np



def plot_trustworthiness(
        df,
        *,                             # everything after * is keyword‑only
        text_shift=1,
        min_gap=0.015,
        fontsize=8,
        figsize=(6, 4),
        dpi=300,
        save_path=None,
        palette=None,                  # mapping {embedding: colour} or None
        y_range=None,                  # (ymin, ymax)  e.g. (0.75, 1) or (None, 1)
        legend=True,                   # NEW: show legend for the lines
        **line_kw):                    # extra kwargs forwarded to ax.plot
    """
    Plot trustworthiness curves for each embedding in *df*.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 1. draw curves and record last points -------------------------------
    label_pos = []                                   # (emb, x_last, y_last)
    for emb, grp in df.groupby("Embedding"):
        color = palette.get(emb) if palette and emb in palette else None
        ax.plot(grp["n_neighbors"],
                grp["Trustworthiness"],
                label=emb,
                color=color,
                **line_kw)
        label_pos.append((emb,
                          grp["n_neighbors"].iloc[-1],
                          grp["Trustworthiness"].iloc[-1]))

    # 2. enforce minimum vertical spacing of text labels ------------------
    label_pos.sort(key=lambda t: -t[2])              # highest first
    for i in range(1, len(label_pos)):
        prev_y = label_pos[i-1][2]
        cur_y  = label_pos[i][2]
        if prev_y - cur_y < min_gap:
            label_pos[i] = (label_pos[i][0],
                            label_pos[i][1],
                            prev_y - min_gap)

    # 3. add labels --------------------------------------------------------
    for emb, x_last, y_adj in label_pos:
        ax.text(x_last + text_shift, y_adj, emb,
                fontsize=fontsize, va='center')

    # 4. cosmetics ---------------------------------------------------------
    ax.set_xlim(right=ax.get_xlim()[1] + text_shift * 1.5)

    if y_range is not None:
        ymin, ymax = y_range
        ax.set_ylim(bottom=ymin if ymin is not None else ax.get_ylim()[0],
                    top=ymax  if ymax is not None else ax.get_ylim()[1])

    ax.set_title('Trustworthiness of Latent Embeddings', fontsize=9)
    ax.set_xlabel('Number of Neighbors', fontsize=8)
    ax.set_ylabel('Trustworthiness', fontsize=8)
    ax.tick_params(labelsize=7)

    if legend:
        ax.legend(frameon=False, fontsize=fontsize)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()




def plot_distance_heatmap(distances, n_cols=3, annot_value=False, figsize=(2, 1.6), cbar=True, vmax=None, fontsize=10, rasterize=True, dpi=300, save_path=None):
    """
    Plots heatmaps of pairwise distance matrices in a grid layout.

    Args:
        distances (dict): 
            Dictionary where keys are distance metric names and values are distance matrices.
        n_cols (int, optional): 
            Number of columns in the subplot grid. Defaults to `3`.
        annot_value (bool, optional): 
            Whether to annotate heatmap values. Defaults to `False`.
        figsize (tuple, optional): 
            Base figure size for each subplot (width, height). Defaults to `(2, 1.6)`.
        cbar (bool, optional): 
            Whether to display a color bar. Defaults to `True`.
        fontsize (int, optional): 
            Font size for axis labels and titles. Defaults to `10`.
        rasterize (bool, optional): 
            Whether to rasterize the heatmap for better performance. Defaults to `True`.
        dpi (int, optional): 
            Resolution (dots per inch) for saving the figure. Defaults to `300`.
        save_path (str, optional): 
            File path to save the figure. If `None`, the plot is displayed instead. Defaults to `None`.

    Returns:
        None

    Example:
        ```python
        plot_distance_heatmap(distances, n_cols=4, save_path="distance_heatmaps.png")
        ```
    """
    # Visualize the distance matrices in a more compact layout
    from scipy.spatial.distance import squareform
    import matplotlib.pyplot as plt
    import matplotlib.collections as mcoll
    import seaborn as sns
    import numpy as np
    
    keys = list(distances.keys())
    n_plots = len(keys)
    n_rows = int(np.ceil(n_plots / n_cols))
    base_width = figsize[0]
    base_height = figsize[1]
    
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(base_width * n_cols, base_height * n_rows), dpi=dpi
    )
    axes = np.atleast_2d(axes).flatten() 

    cbar_kws = {"shrink": 0.8, "label": None, "format": "%.2f", "pad": 0.02} if cbar else None

    for i, key in enumerate(keys):
        ax = axes[i]
        hmap = sns.heatmap(
            squareform(distances[key]),
            ax=ax,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            annot=annot_value, fmt=".2f", annot_kws={"size": 6},
            vmax=vmax,
            cbar=cbar,  # Pass the value of cbar here to toggle the color bar
            cbar_kws=cbar_kws
        )

        # Example to iterate over the children of the heatmap and set rasterization
        if rasterize:
            for artist in hmap.get_children():
                if isinstance(artist, mcoll.QuadMesh):
                    artist.set_rasterized(True)
                    
        ax.set_title(key, fontsize=fontsize)  # Increase the title font size

    # Hide empty subplots if n_plots < n_cols * n_rows
    for j in range(n_plots, n_cols * n_rows):
        fig.delaxes(axes[j])

    # Set compact layout
    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.3)  # Adjust padding

    # Save the plot
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_geometry_scatter(data_dict, correlation = None, ground_key='PCA_no_noise', linear_fit = False, s=1, c=None, alpha=0.5, n_cols=3, fontsize=8, figsize=(4, 4), rasterized=True, dpi=300, save_path=None):
    """
    Plots scatter plots comparing geometric properties of embeddings.

    Args:
        data_dict (dict): 
            Dictionary where keys are embedding names and values are distance vectors.
        correlation (pd.DataFrame, optional): 
            DataFrame containing correlation values for each embedding. Defaults to `None`.
        ground_key (str, optional): 
            Key used as the reference ground-truth embedding. Defaults to `'PCA_no_noise'`.
        linear_fit (bool, optional): 
            Whether to fit and plot a linear regression line. Defaults to `False`.
        s (float, optional): 
            Marker size in scatter plots. Defaults to `1`.
        c (str or array-like, optional): 
            Color of points. Defaults to `None`.
        alpha (float, optional): 
            Opacity of points. Defaults to `0.5`.
        n_cols (int, optional): 
            Number of columns in the subplot grid. Defaults to `3`.
        fontsize (int, optional): 
            Font size for axis labels and titles. Defaults to `8`.
        figsize (tuple, optional): 
            Base figure size for each subplot (width, height). Defaults to `(4, 4)`.
        rasterized (bool, optional): 
            Whether to rasterize scatter points for performance. Defaults to `True`.
        dpi (int, optional): 
            Resolution (dots per inch) for saving the figure. Defaults to `300`.
        save_path (str, optional): 
            File path to save the figure. If `None`, the plot is displayed instead. Defaults to `None`.

    Returns:
        None

    Example:
        ```python
        plot_geometry_scatter(data_dict, correlation=correlation_df, save_path="geometry_scatter.png")
        ```
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    keys = list(data_dict.keys())
    n_plots = len(keys)
    n_rows = int(np.ceil(n_plots / n_cols))
    base_width = figsize[0]
    base_height = figsize[1]
    
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(base_width * n_cols, base_height * n_rows), dpi=dpi
    )
    axes = np.atleast_2d(axes).flatten() 
    for i, key in enumerate(keys):
        # avoid plotting empty subplots
        if i >= n_plots:
            break
        ax = axes[i]

        # flat distance[ground_key] to np array if dict
        if isinstance(data_dict[ground_key], dict):
            ground_val = np.array([data_dict[ground_key][k] for k in data_dict[ground_key].keys()])
            latent_val = np.array([data_dict[key][k] for k in data_dict[key].keys()])
        else:        
            ground_val = data_dict[ground_key] 
            latent_val = data_dict[key]

        ax.scatter(ground_val, latent_val, s=s, c=c, alpha=alpha, edgecolors='none', rasterized=rasterized)
        # Perform linear regression
        if linear_fit:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(ground_val.reshape(-1, 1), latent_val)
            ax.plot(ground_val, reg.predict(ground_val.reshape(-1, 1)), color='red', linewidth=1)

        if correlation is not None:
            # Compute avearge correlation across columns
            corr_val = correlation.loc[key, :].mean()
            corr_text = '\n' + f'Corr: {corr_val:.2f}'
        else:
            corr_text = ''
        ax.set_title(f'{key}{corr_text}', fontsize=fontsize)
        ax.set_xlabel(f'{ground_key}', fontsize=fontsize)
        ax.set_ylabel(f'{key}', fontsize=fontsize)
        # ax set tick label font
        ax.tick_params(axis='both', which='major', labelsize=fontsize-1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()



