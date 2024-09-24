
import matplotlib.pyplot as plt
import scanpy as sc
import umap
import warnings
import numpy as np
import seaborn as sns
from .. import logger
import time


def plot_embedding(adata, show_emb, show_cols=None, 
                   pal = 'tab20',
                   highlight_indices=None,
                   highlight_size=20, draw_path=False, alpha=0.9,
                   figsize=(9, 3), dpi=300, ncols=1,
                    font_size=8, point_size=10, legend_loc='on data', save_path=None):
    """
    Plot embeddings from an AnnData object with customizable parameters.

    Parameters:
    adata (anndata.AnnData): The AnnData object.
    show_emb (str): The embedding to plot.
    show_cols (list of str): The columns to color the embeddings by.
    figsize (tuple): The size of the figure.
    dpi (int): The resolution of the figure.
    ncols (int): The number of columns in the subplot grid.
    font_size (int): The font size for titles and labels.
    point_size (int): The size of the points in the plot.

    Returns:
    None
    """

    if show_cols is None or len(show_cols) == 0:
        show_cols = [None]  # Use a single plot without coloring

    # Calculate the number of rows needed
    nrows = int(np.ceil(len(show_cols) / ncols))

    # Create subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
    axs = np.atleast_2d(axs).flatten()  # Ensure axs is a 1D array for easy iteration

    warnings.filterwarnings('ignore')
    for col, ax in zip(show_cols, axs):
        num_categories = len(adata.obs[col].unique())
        palette = sns.color_palette(pal, num_categories)  # tab20 supports up to 20 distinct colors, you can adjust for more categories

        sc.pl.embedding(adata, basis=show_emb, color=col, ax=ax, show=False,
                        legend_loc=legend_loc, legend_fontsize=font_size, size=point_size, alpha=alpha,
                        palette=palette)

        # Highlight selected points
        if highlight_indices is not None:
            highlight_data = adata[highlight_indices, :]
            sc.pl.embedding(highlight_data, basis=show_emb, color=col, ax=ax, show=False,
                            legend_loc=None, legend_fontsize=font_size, size=highlight_size, alpha=1.0)

            if draw_path:
                # Extract the embedding coordinates
                embedding = adata.obsm[show_emb]
                path_coords = embedding[highlight_indices, :]

                # Draw the path
                ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2)  # Red line for the path

        ax.set_title(ax.get_title(), fontsize=font_size)  # Adjust plot title font size
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)  # Adjust X-axis label font size
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)  # Adjust Y-axis label font size
        cbar = ax.collections[-1].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=font_size)

    # Turn off any unused axes
    for ax in axs[len(show_cols):]:
        ax.axis('off')

    #plt.tight_layout()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)

def plot_embedding_3d(adata, embedding_key='encoded_UMAP', color_by='batch', save_path=None, point_size=3,
                           opacity=0.7, width=800, height=600):
    """
    Visualize 3D embedding and color by selected columns in adata.obs, with options for plot aesthetics and saving the plot.

    Parameters:
    adata : AnnData
        AnnData object containing embeddings and metadata.
    embedding_key : str, optional
        Key in adata.obsm where the embedding is stored. Default is 'encoded_UMAP'.
    color_by : str, optional
        Column in adata.obs to color the points by. Default is 'batch'.
    save_path : str, optional
        Path to save the plot. If None, the plot will not be saved.
    point_size : int, optional
        Size of the points in the plot. Default is 3.
    opacity : float, optional
        Opacity of the points in the plot. Default is 0.7.
    width : int, optional
        Width of the plot in pixels. Default is 800.
    height : int, optional
        Height of the plot in pixels. Default is 600.

    Returns:
    fig : plotly.graph_objects.Figure
        Plotly figure object.
    """
    import plotly.express as px
    
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding key '{embedding_key}' not found in adata.obsm")

    if color_by not in adata.obs:
        raise KeyError(f"Column '{color_by}' not found in adata.obs")

    embedding = adata.obsm[embedding_key]

    if embedding.shape[1] < 3:
        raise ValueError(f"Embedding '{embedding_key}' must have at least 3 dimensions")

    # Use only the first 3 dimensions for plotting
    embedding = embedding[:, :3]

    # Create a DataFrame for Plotly
    df = adata.obs.copy()
    df['DIM1'] = embedding[:, 0]
    df['DIM2'] = embedding[:, 1]
    df['DIM3'] = embedding[:, 2]

    # Plotly 3D scatter plot
    fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=color_by,
                        title=f'3D Embedding colored by {color_by}', labels={'color': color_by},
                        opacity=opacity)

    fig.update_traces(marker=dict(size=point_size), selector=dict(mode='markers'))
    fig.update_layout(width=width, height=height)

    if save_path:
        fig.write_html(save_path)
        logger.info(f"3D plot saved to {save_path}")

    fig.show()
    return fig





def plot_top_genes_embedding(adata, ranked_lists, show_emb, top_x=4, figsize=(5, 1.2),
                             dpi=300, font_size=3, point_size=5, legend_loc='on data', save_path=None):
    """
    Plot the expression of top x genes for each neuron on the embedding in a compact way.

    Parameters:
    - adata (anndata.AnnData): The AnnData object.
    - ranked_lists (dict): A dictionary with neuron names as keys and ranked gene lists as values.
    - show_emb (str): The embedding to plot.
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
        show_cols = list(ranked_list['Gene'][0:top_x])
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
            show_emb,
            show_cols=show_cols,
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

