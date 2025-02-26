
import matplotlib.pyplot as plt
import scanpy as sc
import umap
import warnings
import numpy as np
import time
import pandas as pd
import matplotlib.colors as mcolors
from .. import logger
from .palettes import get_color_mapping


def plot_embedding(adata, basis, color_by=None, 
                   pal=None, highlight_indices=None, default_color='lightgrey', highlight_color='black',
                   highlight_size=20, draw_path=False, alpha=0.9, text_alpha=0.5,
                   figsize=(9, 3), dpi=300, ncols=1, ax = None,
                   title=None, xlabel = None, ylabel = None, xticks=True, yticks=True,
                   colorbar_loc='right',
                   vmax_quantile=None, vmax=None,
                   font_size=8, point_size=10, path_width=1, legend_loc='on data', 
                   rasterized=True,
                   seed=42,
                   save_path=None):
    warnings.filterwarnings('ignore')

    if color_by is None or len(color_by) == 0:
        color_by = [None]  # Use a single plot without coloring

    if ax is None:
        nrows = int(np.ceil(len(color_by) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
        axs = np.atleast_2d(axs).flatten()  # Ensure axs is a 1D array for easy iteration
        return_single_ax = False
    else:
        axs = [ax]
        return_single_ax = True
        assert len(axs) == len(color_by), "Number of axes must match the number of color_by columns"

    if not isinstance(pal, dict):
        pal = {col: pal for col in color_by}

    for col, ax in zip(color_by, axs):
        data_col, cmap, palette = get_color_mapping(adata, col, pal, seed=seed)

        # Compute vmax based on the quantile if vmax_quantile is provided and color_by is numeric
        if pd.api.types.is_numeric_dtype(data_col):
            if vmax is not None:
                pass  # Use the provided vmax
            elif vmax_quantile is not None:
                import scipy.sparse as sp
                if col in adata.var_names:  # If color_by is a gene
                    expression_values = adata[:, col].X
                    if sp.issparse(expression_values):
                        expression_values = expression_values.toarray().flatten()
                    vmax = np.percentile(expression_values, vmax_quantile * 100)
                elif col in adata.obs:  # If color_by is in adata.obs
                    vmax = np.percentile(data_col, vmax_quantile * 100)
                else:
                    raise ValueError(f"Unknown column '{col}' in adata")
            else:
                vmax = data_col.max()            

        if col is None:
            sc.pl.embedding(adata, basis=basis, ax=ax, show=False,
                                 legend_loc='right margin', legend_fontsize=font_size,
                                 size=point_size, alpha=alpha)
            for collection in ax.collections:
                collection.set_color(default_color)
        elif pd.api.types.is_numeric_dtype(data_col):
            sc.pl.embedding(adata, basis=basis, color=col, ax=ax, show=False,
                            legend_loc='right margin', legend_fontsize=font_size,
                            size=point_size, alpha=alpha, cmap=cmap, colorbar_loc=colorbar_loc,
                            vmax=vmax  # Use the computed vmax if provided
                            )
        else:
            sc.pl.embedding(adata, basis=basis, color=col, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=font_size,
                            size=point_size, alpha=alpha, palette=palette)

        if legend_loc == 'on data':
            for text in ax.texts:
                text.set_alpha(text_alpha)

        # Highlight selected points
        if highlight_indices is not None:
            # Extract the coordinates for highlighting
            embedding = adata.obsm[basis]

            if col is None:
                # Highlight without color-by
                ax.scatter(
                    embedding[highlight_indices, 0],
                    embedding[highlight_indices, 1],
                    s=highlight_size,
                    linewidths=0,
                    color=highlight_color,
                    alpha=1.0,
                    zorder=1,  # Ensure points are on top
                )
            elif pd.api.types.is_numeric_dtype(data_col):
                # Highlight with numeric color mapping
                if highlight_color is None:
                    colors = data_col.iloc[highlight_indices]
                    norm = plt.Normalize(vmin=data_col.min(), vmax=data_col.max())
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    highlight_colors = sm.to_rgba(colors)
                else:
                    highlight_colors = highlight_color
                ax.scatter(
                    embedding[highlight_indices, 0],
                    embedding[highlight_indices, 1],
                    s=highlight_size,
                    linewidths=0,
                    color=highlight_colors,
                    alpha=1.0,
                    zorder=1,
                )
            else:
                # Highlight with categorical color mapping
                if highlight_color is None:
                    colors = data_col.iloc[highlight_indices].map(palette)
                else:
                    colors = highlight_color
                ax.scatter(
                    embedding[highlight_indices, 0],
                    embedding[highlight_indices, 1],
                    s=highlight_size,
                    linewidths=0,
                    color=colors,
                    alpha=1.0,
                    zorder=1,
                )

            if draw_path:
                # Draw path through highlighted points
                path_coords = embedding[highlight_indices, :]
                ax.plot(
                    path_coords[:, 0],
                    path_coords[:, 1],
                    'r-', linewidth=path_width, alpha=alpha, zorder=2
                )

        ax.set_title(ax.get_title() if title is None else title, fontsize=font_size)
        ax.set_xlabel('' if xlabel is None else xlabel, fontsize=font_size-2)
        ax.set_ylabel('' if ylabel is None else ylabel, fontsize=font_size-2)
        ax.set_xticks([]) if not xticks else None
        ax.set_yticks([]) if not yticks else None

        if hasattr(ax, 'collections') and len(ax.collections) > 0:
            cbar = ax.collections[-1].colorbar
            if cbar is not None:
                cbar.ax.tick_params(labelsize=font_size)

        if rasterized:
            import matplotlib.collections as mcoll
            for artist in ax.get_children():
                if isinstance(artist, mcoll.PathCollection):
                    artist.set_rasterized(True)

    for ax in axs[len(color_by):]:
        ax.axis('off')

    if save_path is not None and not return_single_ax:
        fig.savefig(save_path, dpi=dpi)

    if not return_single_ax:
        plt.show()


# Portal method to choose either plot_embedding_3d_plotly or plot_embedding_3d_matplotlib, given the engine parameter
def plot_embedding_3d(adata, basis='encoded_UMAP', color_by='batch', pal=None, save_path=None, point_size=3, opacity=0.7, seed=42, width=800, height=600, engine='plotly', autosize=True, static=False, static_format='png'):
    if engine == 'plotly':
        return plot_embedding_3d_plotly(adata, basis, color_by, pal, save_path, point_size, opacity, seed, width, height, autosize, static, static_format)
    elif engine == 'matplotlib':
        return plot_embedding_3d_matplotlib(adata, basis, color_by, pal, save_path, point_size, opacity, seed, width, height, static_format=static_format)
    else:
        raise ValueError(f"Unknown engine '{engine}' for 3D embedding plot. Use 'plotly' or 'matplotlib'.")


def plot_embedding_3d_plotly(
        adata, 
        basis='encoded_UMAP', 
        color_by='batch', 
        pal=None,
        save_path=None, 
        point_size=3,
        opacity=0.7, 
        seed=42, 
        width=800, 
        height=600,
        autosize=True,
        static=False,                 # <--- New parameter
        static_format='png',          # <--- New parameter
        title=None
    ):

    import numpy as np
    import pandas as pd
    import matplotlib.colors as mcolors
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

    # Ensure pal is a dict keyed by each color column
    if not isinstance(pal, dict):
        pal = {col: pal for col in color_by}

    figs = []
    for col in color_by:
        # Retrieve color mapping
        data_col, cmap, palette = get_color_mapping(adata, col, pal, seed=seed)
        
        # Plot based on data type: numeric or categorical
        if pd.api.types.is_numeric_dtype(data_col):
            colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 256)]
            colorscale = [[i / (len(colors) - 1), color] for i, color in enumerate(colors)]
            fig = px.scatter_3d(
                df, 
                x='DIM1', 
                y='DIM2', 
                z='DIM3', 
                color=col,
                title=title,
                labels={'color': col}, 
                opacity=opacity,
                color_continuous_scale=colorscale
            )
        else:
            fig = px.scatter_3d(
                df, 
                x='DIM1', 
                y='DIM2', 
                z='DIM3', 
                color=col,
                title=title,
                labels={'color': col}, 
                opacity=opacity,
                color_discrete_map=palette
            )

        fig.update_traces(marker=dict(size=point_size))
        if autosize:
            fig.update_layout(autosize=True,height=height)
        else:
            fig.update_layout(width=width, height=height)

        # Save interactive plot if save_path is provided
        if save_path:
            save_path_str = str(save_path)
            # e.g., "my_plot.html" -> "my_plot_color_col.html"
            col_save_path = save_path_str.replace('.html', f'_{col}.html')
            fig.write_html(col_save_path)
            logger.info(f"3D plot saved to {col_save_path}")

            # Save static image if requested
            if static:
                col_save_path_static = save_path_str.replace('.html', f'_{col}.{static_format}')
                print(col_save_path_static)
                fig.write_image(col_save_path_static)
                logger.info(f"Static 3D plot saved to {col_save_path_static}")

        figs.append(fig)
        
        # Show the interactive plot if not saving statically
        # (Or you could show it regardless, depending on your needs)
        if not static:
            fig.show()

    return figs


def plot_embedding_3d_matplotlib(
    adata, 
    basis='encoded_UMAP', 
    color_by='batch', 
    pal=None,
    save_path=None, 
    point_size=3,
    alpha=0.7, 
    marker_style='o',          
    edge_color='none',         
    edge_width=0,              
    seed=42, 
    width=6, 
    height=6,
    dpi=300,
    show_legend=True,

    # Appearance toggles
    title=None,
    show_title=True,
    title_font_size=10,
    show_axis_labels=True,
    axis_label_font_size=8,
    show_ticks=True,
    show_tick_labels=True,
    tick_label_font_size=8,
    show_grid=False,

    # View angle
    elev=30,    
    azim=45,
    zoom_factor=0.5,
    box_aspect_ratio=None,

    # Highlight indices
    highlight_indices=None,
    highlight_color='black',
    highlight_size=20,
    highlight_alpha=1.0,

    # Quantile color for vmax
    vmax_quantile=None,

    # New parameter to rasterize points
    rasterized=False,

    # If you want to plot into an existing axis
    ax=None
    ):
    if basis not in adata.obsm:
        raise KeyError(f"Embedding key '{basis}' not found in adata.obsm")

    embedding = adata.obsm[basis]
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    if embedding.shape[1] < 3:
        raise ValueError(f"Embedding '{basis}' must have at least 3 dimensions")

    df = adata.obs.copy()
    df['DIM1'] = embedding[:, 0]
    df['DIM2'] = embedding[:, 1]
    df['DIM3'] = embedding[:, 2]

    # Get color mapping
    data_col, cmap, palette_dict = get_color_mapping(adata, color_by, pal, seed=seed)

    # Create fig/ax if not provided
    created_new_fig = False
    if ax is None:
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        created_new_fig = True
    else:
        fig = ax.figure

    ax.view_init(elev=elev, azim=azim)
    if box_aspect_ratio is not None:
        ax.set_box_aspect(box_aspect_ratio)

    # Title
    if show_title:
        title_str = title if title else f"3D Embedding colored by '{color_by}'"
        ax.set_title(title_str, fontsize=title_font_size)

    # Convert categorical data to colors
    if pd.api.types.is_numeric_dtype(data_col):
        print("111")
        print(vmax_quantile)
        if vmax_quantile is not None:
            vmax = np.percentile(data_col, vmax_quantile * 100)
            print(f"Using vmax={vmax} based on quantile {vmax_quantile}")
            data_col = np.clip(data_col, 0, vmax)
        colors = data_col
    else:
        colors = data_col.astype('category').map(palette_dict)

    # **Step 1: Plot all points as transparent background (establish depth ordering)**
    ax.scatter(
        df['DIM1'], df['DIM2'], df['DIM3'],
        c='none',  # Invisible, but included for depth sorting
        alpha=0, 
        s=point_size,
        marker=marker_style,
        edgecolors='none',
        rasterized=rasterized,
        zorder=1
    )

    # **Step 2: Plot non-highlighted points**
    if highlight_indices is not None:
        non_highlight_mask = ~df.index.isin(highlight_indices)
    else:
        non_highlight_mask = np.ones(len(df), dtype=bool)

    ax.scatter(
        df.loc[non_highlight_mask, 'DIM1'],
        df.loc[non_highlight_mask, 'DIM2'],
        df.loc[non_highlight_mask, 'DIM3'],
        c=colors[non_highlight_mask],
        alpha=alpha,
        s=point_size,
        marker=marker_style,
        edgecolors=edge_color,
        linewidths=edge_width,
        rasterized=rasterized,
        zorder=2  # Lower than highlights
    )

    # **Step 3: Plot highlighted points last, ensuring they appear on top**
    if highlight_indices is not None:
        ax.scatter(
            df.loc[highlight_indices, 'DIM1'],
            df.loc[highlight_indices, 'DIM2'],
            df.loc[highlight_indices, 'DIM3'],
            c=highlight_color,
            s=highlight_size,
            alpha=highlight_alpha,
            marker=marker_style,
            edgecolors=edge_color,
            linewidths=edge_width,
            rasterized=rasterized,  # Ensure no compression artifacts for highlights
            zorder=3  # Ensures they are plotted last
        )

    # Axis labels
    if show_axis_labels:
        ax.set_xlabel("DIM1", fontsize=axis_label_font_size)
        ax.set_ylabel("DIM2", fontsize=axis_label_font_size)
        ax.set_zlabel("DIM3", fontsize=axis_label_font_size)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

    # Zoom in
    x_min, x_max = df['DIM1'].min(), df['DIM1'].max()
    y_min, y_max = df['DIM2'].min(), df['DIM2'].max()
    z_min, z_max = df['DIM3'].min(), df['DIM3'].max()

    x_range = (x_max - x_min) * zoom_factor
    y_range = (y_max - y_min) * zoom_factor
    z_range = (z_max - z_min) * zoom_factor

    ax.set_xlim([x_min + x_range, x_max - x_range])
    ax.set_ylim([y_min + y_range, y_max - y_range])
    ax.set_zlim([z_min + z_range, z_max - z_range])

    # Grid
    ax.grid(show_grid)

    # Ticks
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Tick labels
    if not show_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    else:
        ax.tick_params(labelsize=tick_label_font_size)

    if created_new_fig:
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved 3D matplotlib plot for '{color_by}' to {save_path}")
        plt.show()

    return fig, ax



def plot_top_genes_embedding(adata, ranked_lists, basis, top_x=4, figsize=(5, 1.2),
                             dpi=300, font_size=3, point_size=5, legend_loc='on data', colorbar_loc='right', 
                             vmax_quantile=None, save_path=None):
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
        print(f"Plotting {neuron_title} on {basis}")
        # Generate a unique file suffix if saving
        if save_path:
            file_suffix = f"{neuron_name}_{time.strftime('%b%d-%H%M')}"
            neuron_save_path = f"{save_path}_{file_suffix}.pdf"
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
            vmax_quantile=vmax_quantile,
            save_path=neuron_save_path, 
            xticks=False, yticks=False,
            xlabel=None, ylabel=None,
            colorbar_loc=colorbar_loc
        )

        # Show the plot title with neuron name
        plt.suptitle(neuron_title, fontsize=font_size + 2)
        plt.show()



import numpy as np
import pandas as pd
import logging
import math
import warnings
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.collections import PathCollection, LineCollection

logger = logging.getLogger(__name__)

def plot_all_embeddings(
    adata,
    combined_keys,
    color_bys=['time', 'batch'],
    basis_types=['PAGA', 'KNN', 'PCA', 'UMAP'],
    pal={'time': 'viridis', 'batch': 'Set1'},
    vmax_quantile=None,  # New parameter for quantile-based vmax calculation
    k=15,
    edges_color='grey',
    edges_width=0.05,
    layout='kk',
    threshold=0.1,
    node_size_scale=0.1,
    edge_width_scale=0.1,
    font_size=7,
    legend_font_size=2,
    point_size=2.5,
    alpha=0.8,
    figsize=(9, 0.9),
    ncols=11,
    seed=42,
    leiden_key='leiden',
    leiden_resolution=1.0,
    legend_loc=None,
    colorbar_loc=None,
    rasterized=True,
    save_dir='.',
    dpi=300,
    save_format='png',
    file_suffix='plot',
    # ------------------------
    # Highlight parameters
    highlight_indices=None,
    highlight_color='black',
    highlight_size=20,
    draw_path=False,
    path_width=1
):
    """
    Plot multiple embeddings (PAGA, KNN, PCA, UMAP) for given keys and colorings.
    Optionally highlight a subset of points (indices). If highlight_color is None,
    highlighted points will use the same color mapping as the rest (numeric or categorical).
    """
    def highlight_points(ax, adata, basis_key, data_col, cmap, palette,
                         highlight_indices, highlight_color, highlight_size,
                         alpha=1.0, path_width=1, draw_path=False):
        """
        Helper to scatter and optionally connect highlight_indices on top of an existing plot.
        If highlight_color is None, use the same color mapping (numeric or categorical).
        """
        if basis_key not in adata.obsm:
            return  # If there's no embedding to highlight, just return

        embedding = adata.obsm[basis_key]
        if len(highlight_indices) == 0:
            return  # Nothing to highlight

        # Decide the colors for highlight points
        if highlight_color is None:
            # Use the same colormap/palette as the main scatter
            if pd.api.types.is_numeric_dtype(data_col):
                # numeric => map highlight points to the same numeric colormap
                import matplotlib as mpl
                norm = mpl.colors.Normalize(vmin=data_col.min(), vmax=data_col.max())
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

                # If data_col is a pd.Series, get the relevant subset
                highlight_values = data_col.iloc[highlight_indices]
                highlight_colors = sm.to_rgba(highlight_values)
            else:
                # categorical => map highlight points with the same palette
                if isinstance(palette, dict):
                    # If palette is a dictionary mapping category -> color
                    highlight_values = data_col.iloc[highlight_indices]
                    highlight_colors = highlight_values.map(palette)
                else:
                    # If palette is just a list or None, fallback to black or some default
                    highlight_colors = 'black'
        else:
            # Use a fixed color
            highlight_colors = highlight_color

        # Plot highlight points
        ax.scatter(
            embedding[highlight_indices, 0],
            embedding[highlight_indices, 1],
            s=highlight_size,
            linewidths=0,
            color=highlight_colors,
            alpha=1.0,
            zorder=5,  # bring to front
            label='highlighted'
        )

        if draw_path and len(highlight_indices) > 1:
            # Connect the highlighted points with a path (optional)
            path_coords = embedding[highlight_indices]
            ax.plot(
                path_coords[:, 0],
                path_coords[:, 1],
                color=highlight_colors[0] if isinstance(highlight_colors, np.ndarray) else highlight_colors,
                linewidth=path_width,
                alpha=alpha,
                zorder=6
            )

    import scipy.sparse as sp

    nrows = int(np.ceil(len(combined_keys) / ncols))

    for basis_type in basis_types:
        for color_by in color_bys:
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
            axs = np.atleast_2d(axs).flatten()  # Flatten for easy iteration

            for key, ax in zip(combined_keys, axs):
                logger.info(f"Plotting {key} with {color_by} in {basis_type}")
                data_col, cmap, palette = get_color_mapping(adata, color_by, pal, seed=seed)

                # Compute vmax based on quantile if numeric
                vmax = None
                if vmax_quantile is not None and pd.api.types.is_numeric_dtype(data_col):
                    if color_by in adata.var_names:  # If color_by is a gene
                        expression_values = adata[:, color_by].X
                        if sp.issparse(expression_values):
                            expression_values = expression_values.toarray().flatten()
                        vmax = np.percentile(expression_values, vmax_quantile * 100)
                    elif color_by in adata.obs:  # numeric column in obs
                        vmax = np.percentile(data_col, vmax_quantile * 100)

                # Determine the embedding/basis name
                if basis_type != '':
                    # e.g. key="latent", basis_type="UMAP" => "latent_UMAP"
                    # if basis_type already in key, use key as-is
                    basis = f'{key}_{basis_type}' if basis_type not in key else key
                else:
                    basis = key

                # ============ PCA or UMAP or direct obsm-based embeddings ============ #
                if basis_type == '' or basis_type=='PCA' or 'UMAP' in basis:
                    if basis not in adata.obsm:
                        # If this basis doesn't exist, show empty axis
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                        ax.set_title(key, fontsize=font_size)
                        ax.set_xlabel('')
                        ax.set_ylabel('')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    # Main scatter with sc.pl.embedding
                    if pd.api.types.is_numeric_dtype(data_col):
                        sc.pl.embedding(
                            adata, basis=basis, color=color_by, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=legend_font_size,
                            size=point_size, alpha=alpha, cmap=cmap, colorbar_loc=colorbar_loc,
                            vmax=vmax
                        )
                    else:
                        sc.pl.embedding(
                            adata, basis=basis, color=color_by, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=legend_font_size,
                            size=point_size, alpha=alpha, palette=palette
                        )

                    # Highlight indices on top
                    if highlight_indices is not None:
                        highlight_points(
                            ax, adata, basis, data_col, cmap, palette,
                            highlight_indices, highlight_color,
                            highlight_size, alpha=alpha, path_width=path_width, draw_path=draw_path
                        )

                # ============ KNN ============ #
                elif basis_type == 'KNN':
                    # Recompute neighbors => can overwrite existing info, be mindful
                    sc.pp.neighbors(adata, n_neighbors=k, use_rep=key, random_state=seed)
                    sc.tl.draw_graph(adata, layout=layout, random_state=seed)
                    if pd.api.types.is_numeric_dtype(data_col):
                        sc.pl.draw_graph(
                            adata, color=color_by, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=legend_font_size,
                            size=point_size, alpha=alpha, cmap=cmap, edges=True,
                            edges_width=edges_width, edges_color=edges_color,
                            colorbar_loc=colorbar_loc, vmax=vmax
                        )
                    else:
                        sc.pl.draw_graph(
                            adata, color=color_by, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=legend_font_size,
                            size=point_size, alpha=alpha, palette=palette,
                            edges=True, edges_width=edges_width, edges_color=edges_color
                        )

                    # If we want to highlight the same cells, we use the layout in adata.obsm['X_draw_graph_{layout}']
                    draw_key = f'X_draw_graph_{layout}'
                    if highlight_indices is not None and draw_key in adata.obsm:
                        highlight_points(
                            ax, adata, draw_key, data_col, cmap, palette,
                            highlight_indices, highlight_color,
                            highlight_size, alpha=alpha, path_width=path_width, draw_path=draw_path
                        )

                # ============ PAGA ============ #
                elif basis_type == 'PAGA':
                    sc.pp.neighbors(adata, n_neighbors=k, use_rep=key, random_state=seed)
                    sc.tl.leiden(adata, key_added=leiden_key, resolution=leiden_resolution, random_state=seed)
                    try:
                        sc.tl.paga(adata, groups=leiden_key, use_rna_velocity=False)
                        if pd.api.types.is_numeric_dtype(data_col):
                            sc.pl.paga(
                                adata, threshold=threshold, color=color_by, ax=ax, show=False,
                                layout=layout, fontsize=2, cmap=cmap,
                                node_size_scale=node_size_scale, edge_width_scale=edge_width_scale,
                                colorbar=False
                            )
                        else:
                            sc.pl.paga(
                                adata, threshold=threshold, color=color_by, ax=ax, show=False,
                                layout=layout, fontsize=2, cmap=cmap,
                                node_size_scale=node_size_scale, edge_width_scale=edge_width_scale
                            )
                        # Note: PAGA is cluster-level, so highlighting single cells is non-trivial.
                        # If you need cell-level coords, see sc.pl.paga_compare or custom embedding.

                    except Exception as e:
                        logger.error(f"Error plotting PAGA for {key}: {e}")

                # Simplify title
                if 'PCA' in key:
                    plot_title = key.replace('PCA_', '')
                else:
                    plot_title = key

                ax.set_title(plot_title, fontsize=font_size)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticks([])
                ax.set_yticks([])

                # Rasterize
                if rasterized:
                    for artist in ax.get_children():
                        if isinstance(artist, PathCollection):
                            artist.set_rasterized(True)
                        if isinstance(artist, LineCollection):
                            artist.set_rasterized(True)

            # Hide any leftover axes (if combined_keys < nrows*ncols)
            for ax in axs[len(combined_keys):]:
                ax.set_visible(False)

            # Save figure
            output_path = f"{save_dir}/all_latent_{color_by}_{basis_type}_{file_suffix}.{save_format}"
            plt.savefig(output_path, bbox_inches=None)
            plt.show()




def plot_all_embeddings_3d(
    adata,
    combined_keys,
    color_bys=('time', 'batch'),
    basis_types=('UMAP_3D',),
    pal=None,
    point_size=2.5,
    alpha=0.8,
    figsize=(10, 5),
    ncols=4,
    seed=42,
    legend_font_size=5,
    rasterized=False,
    save_dir='.',
    dpi=300,
    save_format='png',
    file_suffix='3d_plot',
    # Additional default 3D plot aesthetics
    elev=30,
    azim=45,
    zoom_factor=0.0,
    # **kwargs to forward to plot_embedding_3d_matplotlib
    **kwargs
):
    """
    Plots multiple 3D embeddings (stored in adata.obsm) with coloring by different obs/var columns.
    For each combination of basis_type and color_by, we create a multi-panel figure of size (nrows, ncols).

    Parameters
    ----------
    adata : AnnData
        Contains the data and obsm embeddings.
    combined_keys : list
        A list of 'keys' for which we have e.g. 'key_UMAP_3D' in adata.obsm.
    color_bys : tuple or list of str
        The obs/var columns or gene names to color by.
    basis_types : tuple or list of str
        The suffix or full name of the 3D embedding in adata.obsm, e.g. 'UMAP_3D', 'TSNE_3D'.
    pal : dict or None
        Color palette dictionary or fallback.
    vmax_quantile : float or None
        If set, percentile to clamp numeric data at (e.g. 0.99 for 99th percentile).
    point_size : float
        Marker area (points^2) for scatter.
    alpha : float
        Marker opacity.
    figsize : tuple
        Figure size in inches (width, height).
    ncols : int
        Number of columns in the subplot layout.
    seed : int
        Random seed for color mapping.
    legend_font_size : int
        Font size for legend (if categorical data).
    rasterized : bool
        If True, points will be rasterized in the subplots.
    save_dir : str
        Directory to save the figures.
    dpi : int
        Resolution (dots per inch) for saved figures.
    save_format : str
        Output image format, e.g. 'png', 'pdf', 'svg'.
    file_suffix : str
        Suffix added to the saved file name.
    elev : float
        Elevation angle for 3D view.
    azim : float
        Azimuth angle for 3D view.
    **kwargs : 
        Additional parameters passed directly to `plot_embedding_3d_matplotlib`, 
        allowing customization (e.g. show_axis_labels, show_ticks, etc.).

    Returns
    -------
    None
        Saves one figure per (basis_type, color_by) combination.
    """
    import math
    import numpy as np

    if pal is None:
        pal = {'time': 'viridis', 'batch': 'Set1'}

    nrows = math.ceil(len(combined_keys) / ncols)

    for basis_type in basis_types:
        for color_by in color_bys:
            fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
            axs = []

            # Create subplots
            for i in range(len(combined_keys)):
                ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
                axs.append(ax)

            # For each key, we have one subplot (ax)
            for key, ax in zip(combined_keys, axs):
                logger.info(f"Plotting 3D: {key}, color by {color_by}, basis: {basis_type}")

                # Figure out the adata.obsm key
                if basis_type not in key:
                    basis = f"{key}_{basis_type}"
                else:
                    basis = key

                if basis not in adata.obsm:
                    ax.set_title(f"{key} (missing {basis_type})", fontsize=legend_font_size)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                    continue

                embedding_3d = adata.obsm[basis]
                if embedding_3d.shape[1] < 3:
                    ax.set_title(f"{basis} is not 3D", fontsize=legend_font_size)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                    continue


                # Call plot_embedding_3d_matplotlib, passing the existing ax and the additional kwargs
                # Notice we pass in `rasterized=rasterized` and any user extras via **kwargs
                plot_embedding_3d_matplotlib(
                    adata=adata,
                    basis=basis,
                    color_by=color_by,
                    pal=pal,
                    point_size=point_size,
                    alpha=alpha,
                    seed=seed,
                    title=None,              # We'll override the subplot title ourselves
                    show_title=False,        # We do not want the default title
                    marker_style='.',
                    edge_color='none',
                    edge_width=0,
                    elev=elev,
                    azim=azim,
                    zoom_factor=zoom_factor,
                    rasterized=rasterized,
                    ax=ax,
                    **kwargs  # pass all other custom aesthetics
                )

                ax.set_title(key, fontsize=legend_font_size)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

            # Hide leftover axes if we don't fill the grid
            leftover = len(axs) - len(combined_keys)
            if leftover > 0:
                for ax in axs[-leftover:]:
                    ax.set_visible(False)

            # Save figure
            out_fn = f"{save_dir}/all_latent_3D_{color_by}_{basis_type}_{file_suffix}.{save_format}"
            plt.savefig(out_fn, bbox_inches='tight')
            plt.show()
            logger.info(f"Saved multi-panel 3D figure: {out_fn}")





def plot_rotating_embedding_3d_to_mp4(adata, embedding_key='encoded_UMAP', color_by='batch', save_path='rotation.mp4', pal=None,
                                      point_size=3, opacity=0.7, width=800, height=1200, rotation_duration=10, num_steps=60,
                                      legend_itemsize=100, font_size=16, seed=42):
    """
    Visualize a rotating 3D embedding and save it as an MP4 video for presentation purposes.

    Parameters:
    adata : AnnData
        AnnData object containing embeddings and metadata.
    embedding_key : str, optional
        Key in adata.obsm where the embedding is stored. Default is 'encoded_UMAP'.
    color_by : str, optional
        Column in adata.obs to color the points by. Default is 'batch'.
    save_path : str, optional
        Path to save the video. Default is 'rotation.mp4'.
    point_size : int, optional
        Size of the points in the plot. Default is 3.
    opacity : float, optional
        Opacity of the points in the plot. Default is 0.7.
    width : int, optional
        Width of the plot in pixels. Default is 800.
    height : int, optional
        Height of the plot in pixels. Default is 600.
    rotation_duration : int, optional
        Duration of the rotation in seconds. Default is 10 seconds.
    num_steps : int, optional
        Number of steps for the rotation animation. Default is 60.

    Returns:
    None
    """
    import numpy as np
    import plotly.graph_objs as go
    import plotly.express as px
    import moviepy.editor as mpy
    import os
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

    # Create initial 3D scatter plot
    data_col, cmap, palette = get_color_mapping(adata, color_by, pal, seed=seed)
        
    # Plot based on data type: numeric or categorical
    if pd.api.types.is_numeric_dtype(data_col):
        colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 256)]
        colorscale = [[i / (len(colors) - 1), color] for i, color in enumerate(colors)]
        fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=color_by,
                            title=f'3D Embedding colored by {color_by}',
                            labels={'color': color_by}, opacity=opacity,
                            color_continuous_scale=colorscale)
    else:
        fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=color_by,
                            title=f'3D Embedding colored by {color_by}',
                            labels={'color': color_by}, opacity=opacity,
                            color_discrete_map=palette)
        
        # Increase size of the points in the legend
        fig.update_layout(
            legend=dict(
                font=dict(size=font_size),  # Increase legend font size
                itemsizing='constant',  # Make legend items the same size
                itemwidth=max(legend_itemsize, 30)  # Increase legend item width
            )
        )
    
    # fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=color_by,
    #                     title=f'3D Embedding colored by {color_by}', labels={'color': color_by},
    #                     opacity=opacity)

    fig.update_traces(marker=dict(size=point_size), selector=dict(mode='markers'))
    fig.update_layout(width=width, height=height)
    

    # Create rotation steps by defining camera positions
    asp_ratio = 1.5
    def generate_camera_angles(num_steps):
        return [
            dict(
                eye=dict(x=np.cos(2 * np.pi * t / num_steps) * asp_ratio, y=np.sin(2 * np.pi * t / num_steps) * asp_ratio, z=asp_ratio / 2)
            )
            for t in range(num_steps)
        ]

    # Generate camera angles for the rotation
    camera_angles = generate_camera_angles(num_steps)
    print(f"number of frames: {len(camera_angles)}")
    # Save the frames as images
    frame_files = []
    for i, camera_angle in enumerate(camera_angles):
        fig.update_layout(scene_camera=camera_angle)
        frame_file = f"frame_{i:04d}.png"
        fig.write_image(frame_file)
        frame_files.append(frame_file)

    # Create the video using MoviePy
    clips = [mpy.ImageClip(frame).set_duration(rotation_duration / num_steps) for frame in frame_files]
    video = mpy.concatenate_videoclips(clips, method="compose")
    video.write_videofile(save_path, fps=num_steps / rotation_duration)

    # Clean up the temporary image files
    for frame_file in frame_files:
        os.remove(frame_file)

    print(f"Rotation video saved to {save_path}")
