
import matplotlib.pyplot as plt
import scanpy as sc
import umap
import warnings
import numpy as np
import seaborn as sns
import time
import pandas as pd
import matplotlib.colors as mcolors
from .. import logger

NUMERIC_PALETTES = {
    "BlueGreenRed": ["midnightblue", "dodgerblue", "seagreen", "#00C000", "#EEC900", "#FF7F00", "#FF0000"],
    "RdOgYl": ["#D9D9D9", "red", "orange", "yellow"],
    "grey&red": ["grey", "#b2182b"],
    "blue_green_gold": ["#D9D9D9", "blue", "green", "#FFD200", "gold"],
    "black_red_gold": ["#D9D9D9", "black", "red", "#FFD200"],
    "black_red": ["#D9D9D9", "black", "red"],
    "red_yellow": ["#D9D9D9", "red", "yellow"],
    "black_yellow": ["#D9D9D9", "black", "yellow"],
    "black_yellow_gold": ["#D9D9D9", "black", "yellow", "gold"],
}


def get_factor_color(labels, pal='Set1', permute=True):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    unique_labels = np.unique(labels)
    num_colors = len(unique_labels)

    # Check if pal is a custom palette
    if pal in NUMERIC_PALETTES:
        colors = NUMERIC_PALETTES[pal]
        if len(colors) < num_colors:
            # Extend the palette to match the number of labels
            colors = sns.color_palette(colors, num_colors)
    elif pal in plt.colormaps():
        colors = sns.color_palette(pal, num_colors)
    else:
        try:
            colors = sns.color_palette(pal, num_colors)
        except ValueError:
            # Default to 'Set1' if palette not found
            colors = sns.color_palette('Set1', num_colors)

    if permute:
        np.random.seed(1)
        np.random.shuffle(colors)

    # Convert colors to hex codes
    colors_hex = [mcolors.rgb2hex(color) for color in colors]

    # Convert labels to strings
    unique_labels_str = [str(label) for label in unique_labels]

    color_map = dict(zip(unique_labels_str, colors_hex))
    return color_map


def get_numeric_color(pal='RdYlBu'):
    if pal in NUMERIC_PALETTES:
        colors = NUMERIC_PALETTES[pal]
        cmap = mcolors.LinearSegmentedColormap.from_list(pal, colors)
    elif pal in plt.colormaps():
        cmap = plt.get_cmap(pal)
    else:
        cmap = sns.color_palette(pal, as_cmap=True)
    return cmap


def plot_embedding(adata, basis, color_by=None, 
                   pal='Set1',
                   highlight_indices=None,
                   highlight_size=20, draw_path=False, alpha=0.9, text_alpha=0.5,
                   figsize=(9, 3), dpi=300, ncols=1,
                   font_size=8, point_size=10, legend_loc='on data', save_path=None):
    warnings.filterwarnings('ignore')

    if color_by is None or len(color_by) == 0:
        color_by = [None]  # Use a single plot without coloring

    nrows = int(np.ceil(len(color_by) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
    axs = np.atleast_2d(axs).flatten()  # Ensure axs is a 1D array for easy iteration

    if not isinstance(pal, dict):
        pal = {col: pal for col in color_by}

    for col, ax in zip(color_by, axs):
        data_col = adata.obs[col]

        current_pal = pal.get(col, 'Set1') 

        if pd.api.types.is_numeric_dtype(data_col):
            cmap = get_numeric_color(current_pal)
            sc.pl.embedding(adata, basis=basis, color=col, ax=ax, show=False,
                            legend_loc='right margin', legend_fontsize=font_size,
                            size=point_size, alpha=alpha, cmap=cmap)
        else:
            color_map = get_factor_color(data_col, current_pal)
            categories = data_col.astype('category').cat.categories
            palette = [color_map[cat] for cat in categories]
            sc.pl.embedding(adata, basis=basis, color=col, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=font_size,
                            size=point_size, alpha=alpha, palette=palette)

        if legend_loc == 'on data':
            for text in ax.texts:
                text.set_alpha(text_alpha)

        # Highlight selected points
        if highlight_indices is not None:
            highlight_data = adata[highlight_indices, :]
            if pd.api.types.is_numeric_dtype(data_col):
                sc.pl.embedding(highlight_data, basis=basis, color=col, ax=ax, show=False,
                                legend_loc=None, legend_fontsize=font_size,
                                size=highlight_size, alpha=1.0, cmap=cmap)
            else:
                sc.pl.embedding(highlight_data, basis=basis, color=col, ax=ax, show=False,
                                legend_loc=None, legend_fontsize=font_size,
                                size=highlight_size, alpha=1.0, palette=palette)

            if draw_path:
                embedding = adata.obsm[basis]
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                path_coords = embedding[highlight_indices, :]

                ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2)  # Red line for the path

        ax.set_title(ax.get_title(), fontsize=font_size)  # Adjust plot title font size
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)  # Adjust X-axis label font size
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)  # Adjust Y-axis label font size
        
        if hasattr(ax, 'collections') and len(ax.collections) > 0:
            cbar = ax.collections[-1].colorbar
            if cbar is not None:
                cbar.ax.tick_params(labelsize=font_size)

    for ax in axs[len(color_by):]:
        ax.axis('off')

    #plt.tight_layout()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)

def plot_embedding_3d(adata, basis='encoded_UMAP', color_by='batch', pal='Set1',  
                      save_path=None, point_size=3,
                      opacity=0.7, width=800, height=600):

    import plotly.express as px
    
    if basis not in adata.obsm:
        raise KeyError(f"Embedding key '{basis}' not found in adata.obsm")

    embedding = adata.obsm[basis]
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    if embedding.shape[1] < 3:
        raise ValueError(f"Embedding '{basis}' must have at least 3 dimensions")

    embedding = embedding[:, :3]  # Use only the first 3 dimensions for plotting

    df = adata.obs.copy()
    df['DIM1'] = embedding[:, 0]
    df['DIM2'] = embedding[:, 1]
    df['DIM3'] = embedding[:, 2]

    if isinstance(color_by, str):
        color_by = [color_by]

    if not isinstance(pal, dict):
        pal = {col: pal for col in color_by}

    figs = []
    for col in color_by:
        if col not in df:
            raise KeyError(f"Column '{col}' not found in adata.obs")

        data_col = df[col]

        # Get the palette for the current column
        current_pal = pal.get(col, 'Set1')  # Default to 'Set1' if not specified

        # Check if the column is numeric or categorical
        if pd.api.types.is_numeric_dtype(data_col):
            cmap = get_numeric_color(current_pal)
            colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 256)]
            colorscale = [[i / (len(colors) - 1), color] for i, color in enumerate(colors)]
            fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=col,
                                title=f'3D Embedding colored by {col}',
                                labels={'color': col}, opacity=opacity,
                                color_continuous_scale=colorscale)
        else:
            # Categorical data
            color_map = get_factor_color(data_col, current_pal)
            fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=col,
                                title=f'3D Embedding colored by {col}',
                                labels={'color': col}, opacity=opacity,
                                color_discrete_map=color_map)

        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(width=width, height=height)

        if save_path:
            save_path_str = str(save_path)
            col_save_path = save_path_str.replace('.html', f'_{col}.html')
            fig.write_html(col_save_path)
            logger.info(f"3D plot saved to {col_save_path}")

        figs.append(fig)
        fig.show()



def plot_top_genes_embedding(adata, ranked_lists, basis, top_x=4, figsize=(5, 1.2),
                             dpi=300, font_size=3, point_size=5, legend_loc='on data', save_path=None):
    """
    Plot the expression of top x genes for each neuron on the embedding in a compact way.

    Parameters:
    - adata (anndata.AnnData): The AnnData object.
    - ranked_lists (dict): A dictionary with neuron names as keys and ranked gene lists as values.
    - basis (str): The embedding to plot.
    - top_x (int): Number of top genes to display for each neuron.
    - figsize (tuple): The size of the figure.
    - dpi (int): The resolution of the figure.
    - font_size (int): The font size for titles and labels.
    - point_size (int): The size of the points in the plot.
    - legend_loc (str): The location of the legend.
    - save_path (str): The path to save the figure.

    Returns:
    None
    """

    for neuron_name, ranked_list in ranked_lists.items():
        color_by = list(ranked_list['Gene'][0:top_x])
        neuron_title = f"Top {top_x} genes for {neuron_name}"

        # Generate a unique file suffix if saving
        if save_path:
            file_suffix = f"{neuron_name}_{time.strftime('%b%d-%H%M')}"
            neuron_save_path = f"{save_path}_{file_suffix}.png"
        else:
            neuron_save_path = None

        # Call the plot_embedding function
        plot_embedding(
            adata,
            basis,
            color_by=color_by,
            figsize=figsize,
            dpi=dpi,
            ncols=top_x,
            font_size=font_size,
            point_size=point_size,
            legend_loc=legend_loc,
            save_path=neuron_save_path
        )

        # Show the plot title with neuron name
        plt.suptitle(neuron_title, fontsize=font_size + 2)
        plt.show()



def plot_custom_embeddings_umap(custom_embedding=None, figsize=(5, 5), dpi=300, 
                                font_size=3, point_size=5,  save_path=None):
    if custom_embedding is None:
        raise ValueError("custom_embedding must be provided.")
    
    # Extract embeddings (values) and custom names (index)
    embeddings = custom_embedding.values
    custom_names = custom_embedding.index

    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(embeddings)
    
    # Plot the UMAP embeddings
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=point_size, color='blue')

    # Annotate each point with the corresponding custom name
    for i, custom_name in enumerate(custom_names):
        ax.text(umap_embeddings[i, 0], umap_embeddings[i, 1], custom_name, fontsize=font_size, ha='right')

    ax.set_title('UMAP of Embedding', fontsize=font_size + 2)
    ax.set_xlabel('UMAP Dimension 1', fontsize=font_size + 1)
    ax.set_ylabel('UMAP Dimension 2', fontsize=font_size + 1)
    ax.grid(True)

    # Optionally save the figure
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    # Show the plot
    plt.show()

