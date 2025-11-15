
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
import numpy as np
import time
import pandas as pd
import matplotlib.colors as mcolors
from .. import logger
from .palettes import get_color_mapping
import logging
from matplotlib.collections import PathCollection, LineCollection

logger = logging.getLogger(__name__)


def _limits_from_spec(arr, spec):
    if spec is None:
        return None
    mode, a, b = spec
    if mode == 'q':   # quantiles
        lo, hi = np.quantile(arr, [a, b])
    elif mode == 'abs':  # absolute bounds
        lo, hi = float(a), float(b)
    else:
        raise ValueError("x_zoom/y_zoom must be ('q', qmin, qmax) or ('abs', min, max)")
    return lo, hi

def plot_embedding(adata, basis, color_by=None, 
                   pal=None, 
                   highlight_indices=None, 
                   default_color='lightgrey', 
                   highlight_color='black',
                   highlight_size=20, 
                   highlight_density=False,
                   highlight_alpha=1.0,
                   density_color='viridis',
                   density_levels=5,
                   density_alpha=0.5,
                   draw_path=False, alpha=0.9, text_alpha=0.5,
                   figsize=(9, 3), dpi=300, ncols=1, ax = None,
                   title=None, xlabel = None, ylabel = None, xticks=True, yticks=True,
                   colorbar_loc='right',
                   vmax_quantile=None, vmax=None,
                   font_size=8, point_size=10, path_width=1, legend_loc='on data', 
                   legend_markerscale=1.0,
                   x_zoom=None,  # ('q', 0.45, 0.75) or ('abs', xmin, xmax)
                   y_zoom=None,  # ('q', 0.50, 0.80) or ('abs', ymin, ymax)
                   square_zoom=False,
                   rasterized=True,
                   seed=42,
                   save_path=None):
    
    """
    Plots a 2D embedding (e.g., UMAP, PCA) with optional highlighting and color mapping.

    Args:
        adata (AnnData): Single-cell AnnData object containing embeddings and metadata.
        basis (str): The name of the embedding stored in `adata.obsm` (e.g., `'X_umap'`).
        color_by (str | list, optional): Column(s) in `adata.obs` to color the points by. Defaults to None.
        pal (dict, optional): Color palette mapping category values to colors. Defaults to None.
        highlight_indices (list, optional): Indices of points to highlight. Defaults to None.
        default_color (str, optional): Default color for uncolored points. Defaults to "lightgrey".
        highlight_color (str, optional): Color for highlighted points. Defaults to "black".
        highlight_size (int, optional): Size of highlighted points. Defaults to 20.
        draw_path (bool, optional): Whether to draw a path connecting highlighted points. Defaults to False.
        alpha (float, optional): Opacity of points. Defaults to 0.9.
        text_alpha (float, optional): Opacity of text labels. Defaults to 0.5.
        figsize (tuple, optional): Figure size (width, height). Defaults to (9, 3).
        dpi (int, optional): Resolution of the figure. Defaults to 300.
        ncols (int, optional): Number of columns for subplots. Defaults to 1.
        ax (matplotlib.axes.Axes, optional): Axes object for the plot. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        xlabel (str, optional): Label for X-axis. Defaults to None.
        ylabel (str, optional): Label for Y-axis. Defaults to None.
        xticks (bool, optional): Whether to show X-axis ticks. Defaults to True.
        yticks (bool, optional): Whether to show Y-axis ticks. Defaults to True.
        colorbar_loc (str, optional): Location of colorbar ("right", "left", "bottom", etc.). Defaults to "right".
        vmax_quantile (float, optional): If provided, scales the color range to this quantile. Defaults to None.
        vmax (float, optional): Maximum value for color scaling. Defaults to None.
        font_size (int, optional): Font size for annotations. Defaults to 8.
        point_size (int, optional): Size of scatter plot points. Defaults to 10.
        path_width (int, optional): Width of path lines (if `draw_path=True`). Defaults to 1.
        legend_loc (str, optional): Location of the legend ("on data", "right margin", etc.). Defaults to "on data".
        rasterized (bool, optional): If True, rasterizes the points for efficient plotting. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        save_path (str, optional): Path to save the figure. Defaults to None.

    Returns:
        None
    """

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

        # Start clean: determine a local vmax per color_by
        local_vmax = None

        if pd.api.types.is_numeric_dtype(data_col):
            if vmax is not None:
                local_vmax = vmax  # If user manually specified a vmax globally
            elif vmax_quantile is not None:
                import scipy.sparse as sp
                if col in adata.var_names:  # If color_by is a gene
                    expression_values = adata[:, col].X
                    if sp.issparse(expression_values):
                        expression_values = expression_values.toarray().flatten()
                    local_vmax = np.percentile(expression_values, vmax_quantile * 100)
                    local_vmax = max(0.1, local_vmax)
                elif col in adata.obs:
                    local_vmax = np.percentile(data_col, vmax_quantile * 100)
                else:
                    raise ValueError(f"Unknown column '{col}' in adata")
            else:
                local_vmax = data_col.max()  

        with plt.rc_context({'legend.markerscale': legend_markerscale}):
            if col is None:
                sc.pl.embedding(adata, basis=basis, ax=ax, show=False,
                                    legend_loc='right margin', legend_fontsize=font_size,
                                    size=point_size, alpha=alpha, zorder=1)
                for collection in ax.collections:
                    collection.set_color(default_color)
            elif pd.api.types.is_numeric_dtype(data_col):
                if col in adata.var_names:
                    sc.pl.embedding(adata, basis=basis, color=col, ax=ax, show=False,
                                    legend_loc='right margin', legend_fontsize=font_size,
                                    size=point_size, alpha=alpha, cmap=cmap, colorbar_loc=colorbar_loc,
                                    vmin=0, vmax=local_vmax,
                                    zorder=1)
                else:
                    sc.pl.embedding(adata, basis=basis, color=col, ax=ax, show=False,
                                    legend_loc='right margin', legend_fontsize=font_size,
                                    size=point_size, alpha=alpha, cmap=cmap, colorbar_loc=colorbar_loc,
                                    vmax=local_vmax,
                                    zorder=1)
            else:
                sc.pl.embedding(adata, basis=basis, color=col, ax=ax, show=False,
                                legend_loc=legend_loc, legend_fontsize=font_size,
                                size=point_size, alpha=alpha, palette=palette, zorder=1)

        if legend_loc == 'on data':
            for text in ax.texts:
                text.set_alpha(text_alpha)

        # Highlight selected points
        if highlight_indices is not None:
            # Extract the coordinates for highlighting
            embedding = adata.obsm[basis]
            highlight_x = embedding[highlight_indices, 0]
            highlight_y = embedding[highlight_indices, 1]

            if col is None:
                # Highlight without color-by
                ax.scatter(
                    highlight_x,
                    highlight_y,
                    s=highlight_size,
                    linewidths=0,
                    color=highlight_color,
                    alpha=highlight_alpha,
                    zorder=2,  # Ensure points are on top
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
                    highlight_x,
                    highlight_y,
                    s=highlight_size,
                    linewidths=0,
                    color=highlight_colors,
                    alpha=highlight_alpha,
                    zorder=2,
                )
            else:
                # Highlight with categorical color mapping
                if highlight_color is None:
                    colors = data_col.iloc[highlight_indices].map(palette)
                else:
                    colors = highlight_color
                ax.scatter(
                    highlight_x,
                    highlight_y,
                    s=highlight_size,
                    linewidths=0,
                    color=colors,
                    alpha=highlight_alpha,
                    zorder=2,
                )

            if draw_path:
                # Draw path through highlighted points
                path_coords = embedding[highlight_indices, :]
                ax.plot(
                    path_coords[:, 0],
                    path_coords[:, 1],
                    'r-', linewidth=path_width, alpha=alpha, zorder=3
                )

            if highlight_density:
                import seaborn as sns
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                # Plot density within xlim and ylim
                sns.kdeplot(
                    x=highlight_x,
                    y=highlight_y,
                    ax=ax,
                    cmap=density_color,
                    alpha=density_alpha,
                    fill=True,
                    levels=density_levels,
                    zorder=0)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                


        ax.set_title(ax.get_title() if title is None else title, fontsize=font_size)
        ax.set_xlabel('' if xlabel is None else xlabel, fontsize=font_size-2)
        ax.set_ylabel('' if ylabel is None else ylabel, fontsize=font_size-2)
        ax.set_xticks([]) if not xticks else None
        ax.set_yticks([]) if not yticks else None

        # Zoom in utilities
        emb = adata.obsm[basis]
        x = emb[:, 0]
        y = emb[:, 1]

        xlim = _limits_from_spec(x, x_zoom)
        ylim = _limits_from_spec(y, y_zoom)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if square_zoom and (xlim is not None or ylim is not None):
            # enforce a square view box centered on the current limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            cx = 0.5 * (cur_xlim[0] + cur_xlim[1])
            cy = 0.5 * (cur_ylim[0] + cur_ylim[1])
            half = max(cur_xlim[1]-cur_xlim[0], cur_ylim[1]-cur_ylim[0]) / 2.0
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

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
def plot_embedding_3d(adata, basis='encoded_UMAP', color_by='batch', pal=None, save_path=None, 
                      point_size=3, opacity=0.7, seed=42, width=800, height=600, 
                      view_azim=None, view_elev=None, view_dist=None,
                      engine='plotly', autosize=True, static=False, static_format='png'):
    """
    Plots a 3D embedding using Plotly or Matplotlib.

    Args:
        adata (AnnData): Single-cell AnnData object containing embeddings.
        basis (str, optional): The name of the 3D embedding stored in `adata.obsm`. Defaults to `'encoded_UMAP'`.
        color_by (str, optional): Column in `adata.obs` used to color points. Defaults to `'batch'`.
        pal (dict, optional): Color palette mapping categorical variables to colors. Defaults to None.
        save_path (str, optional): Path to save the figure. Defaults to None.
        point_size (int, optional): Size of the points in the plot. Defaults to 3.
        opacity (float, optional): Opacity of the points. Defaults to 0.7.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        width (int, optional): Width of the plot in pixels. Defaults to 800.
        height (int, optional): Height of the plot in pixels. Defaults to 600.
        engine (str, optional): Rendering engine (`'plotly'` or `'matplotlib'`). Defaults to `'plotly'`.
        autosize (bool, optional): Whether to automatically adjust plot size. Defaults to True.
        static (bool, optional): If True, saves the plot as a static image. Defaults to False.
        static_format (str, optional): Format for static image (e.g., `'png'`, `'pdf'`). Defaults to `'png'`.

    Returns:
        plotly.Figure or matplotlib.Figure: A 3D scatter plot.

    Raises:
        ValueError: If the engine is not `'plotly'` or `'matplotlib'`.
    """
    if engine == 'plotly':
        return plot_embedding_3d_plotly(adata, basis, color_by, pal, save_path, point_size, opacity, seed, width, height, 
                                        view_azim=view_azim, view_elev=view_elev, view_dist=view_dist,
                                        autosize=autosize, static=static, static_format=static_format)
    elif engine == 'matplotlib':
        return plot_embedding_3d_matplotlib(adata, basis, color_by, pal, save_path, point_size, opacity, seed, width, height, 
                                            azim=view_azim, elev=view_elev, zoom_factor=view_dist)
    else:
        raise ValueError(f"Unknown engine '{engine}' for 3D embedding plot. Use 'plotly' or 'matplotlib'.")


def _camera_from_angles(azim_deg: float, elev_deg: float, r: float = 1.25):
    """Convert Matplotlib-style (azim, elev) to Plotly camera eye coordinates."""
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return dict(eye=dict(x=x, y=y, z=z), up=dict(x=0, y=0, z=1))

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
        title=None,
        view_azim=None,      # e.g., 290
        view_elev=None,      # e.g., 75
        view_dist=1.25,
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

        if view_azim is not None and view_elev is not None:
            cam = _camera_from_angles(view_azim, view_elev, r=view_dist)
            fig.update_layout(scene_camera=cam)

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
    legend_font_size=8,
    legend_orientation='vertical',
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

    # if not isinstance(embedding, np.ndarray):
    #     embedding = np.array(embedding)
    if embedding.shape[1] < 3:
        raise ValueError(f"Embedding '{basis}' must have at least 3 dimensions")
    
    # Convert embedding to pandas DataFrame and match with adata.obs
    embedding = pd.DataFrame(embedding[:, :3], columns=['DIM1', 'DIM2', 'DIM3'])
    embedding.index = adata.obs.index

    df = adata.obs.copy()
    df['DIM1'] = embedding['DIM1']
    df['DIM2'] = embedding['DIM2']
    df['DIM3'] = embedding['DIM3']

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
    print("vmax_quantile:", vmax_quantile)
    if pd.api.types.is_numeric_dtype(data_col):
        if vmax_quantile is not None:
            vmax = np.nanpercentile(data_col, vmax_quantile * 100)
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
        cmap=cmap,
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
            cmap=cmap,
            s=highlight_size,
            alpha=highlight_alpha,
            marker=marker_style,
            edgecolors=edge_color,
            linewidths=edge_width,
            rasterized=rasterized,  # Ensure no compression artifacts for highlights
            zorder=3  # Ensures they are plotted last
        )

    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    # Build legend/colorbar if requested
    if show_legend:
        if pd.api.types.is_numeric_dtype(data_col):
            # ---- NUMERIC: horizontal/vertical colorbar ----
            arr = np.asarray(data_col)  # robust to Series or ndarray

            if vmax_quantile is not None:
                vmax_used = np.nanpercentile(arr, vmax_quantile * 100.0)
                vmin_used = 0.0  # for gene expr; change if signed values possible
            else:
                vmin_used = np.nanmin(arr)
                vmax_used = np.nanmax(arr)
                if np.isfinite(vmin_used) and np.isfinite(vmax_used) and vmin_used == vmax_used:
                    vmin_used = 0.0
                    vmax_used = max(1.0, vmax_used)

            norm = Normalize(vmin=vmin_used, vmax=vmax_used)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            if legend_orientation == 'horizontal':
                # top horizontal colorbar
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                                    fraction=0.046, pad=0.10)
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.xaxis.set_label_position('top')
            else:
                # default vertical colorbar
                cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)

            cbar.ax.tick_params(labelsize=tick_label_font_size)

        else:
            # ---- CATEGORICAL: legend handles ----
            categories = pd.Categorical(data_col).categories
            handles = [
                Line2D([0], [0], marker='o', linestyle='',
                    markerfacecolor=palette_dict.get(cat, 'gray'),
                    markeredgecolor='none', markersize=6, label=str(cat))
                for cat in categories
            ]

            if legend_orientation == 'horizontal':
                ax.legend(handles=handles,
                        loc='upper center',
                        bbox_to_anchor=(0.5, 1.15),
                        ncol=len(categories) if len(categories) > 0 else 1,
                        fontsize=legend_font_size,
                        frameon=False)
            else:
                ax.legend(handles=handles, loc='best',
                        fontsize=legend_font_size, frameon=False)
                
            
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
    Plots multiple 2D embeddings (PAGA, KNN, PCA, UMAP) with different color mappings.

    Args:
        adata (AnnData): Single-cell AnnData object containing embeddings.
        combined_keys (list): List of feature representations (e.g., `['X_pca', 'X_umap']`).
        color_bys (tuple, optional): List of `adata.obs` columns to color by. Defaults to `("time", "batch")`.
        basis_types (tuple, optional): Types of embeddings to plot. Defaults to `("PAGA", "KNN", "PCA", "UMAP")`.
        pal (dict, optional): Color palettes for each `color_by` variable. Defaults to `{"time": "viridis", "batch": "Set1"}`.
        vmax_quantile (float, optional): Upper quantile for color scaling in numeric data. Defaults to None.
        k (int, optional): Number of neighbors for KNN and PAGA graphs. Defaults to 15.
        edges_color (str, optional): Color of edges in KNN/PAGA graphs. Defaults to `"grey"`.
        edges_width (float, optional): Width of edges in KNN/PAGA graphs. Defaults to 0.05.
        layout (str, optional): Graph layout algorithm for KNN/PAGA. Defaults to `"kk"`.
        threshold (float, optional): Edge threshold for PAGA visualization. Defaults to 0.1.
        node_size_scale (float, optional): Scale factor for PAGA node sizes. Defaults to 0.1.
        edge_width_scale (float, optional): Scale factor for PAGA edge widths. Defaults to 0.1.
        font_size (int, optional): Font size for plot annotations. Defaults to 7.
        legend_font_size (int, optional): Font size for legends. Defaults to 2.
        point_size (float, optional): Size of scatter plot points. Defaults to 2.5.
        alpha (float, optional): Transparency of points. Defaults to 0.8.
        figsize (tuple, optional): Figure size (width, height). Defaults to `(9, 0.9)`.
        ncols (int, optional): Number of columns for subplot grid. Defaults to 11.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        leiden_key (str, optional): Key for Leiden clustering in PAGA. Defaults to `"leiden"`.
        leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
        legend_loc (str, optional): Location of the legend. Defaults to None.
        colorbar_loc (str, optional): Location of the colorbar. Defaults to None.
        rasterized (bool, optional): Whether to rasterize the plot. Defaults to True.
        save_dir (str, optional): Directory to save plots. Defaults to `"."`.
        dpi (int, optional): Image resolution. Defaults to 300.
        save_format (str, optional): Image format (`"png"`, `"pdf"`, etc.). Defaults to `"png"`.
        file_suffix (str, optional): Filename suffix. Defaults to `"plot"`.
        highlight_indices (list, optional): Indices of highlighted points. Defaults to None.
        highlight_color (str, optional): Color of highlighted points. Defaults to `"black"`.
        highlight_size (int, optional): Size of highlighted points. Defaults to 20.
        draw_path (bool, optional): Whether to draw a connecting path for highlights. Defaults to False.
        path_width (int, optional): Width of connecting paths. Defaults to 1.

    Returns:
        None
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
    Plots multiple 3D embeddings with different color mappings across various embedding types.

    Each subplot represents a different embedding (e.g., UMAP_3D) with a specified coloring 
    (e.g., time, batch). This function generates a grid of 3D scatter plots.

    Args:
        adata (AnnData): 
            Single-cell AnnData object containing embeddings.
        combined_keys (list): 
            List of feature representations for which embeddings exist in `adata.obsm`.
        color_bys (tuple of str, optional): 
            List of `adata.obs` columns to color points by. Defaults to `('time', 'batch')`.
        basis_types (tuple of str, optional): 
            Types of embeddings to plot. Defaults to `('UMAP_3D',)`.
        pal (dict, optional): 
            Dictionary mapping categorical values to colors. Defaults to None.
        point_size (float, optional): 
            Size of points in the scatter plot. Defaults to `2.5`.
        alpha (float, optional): 
            Opacity of points. Defaults to `0.8`.
        figsize (tuple, optional): 
            Figure size in inches (width, height). Defaults to `(10, 5)`.
        ncols (int, optional): 
            Number of columns in the subplot grid. Defaults to `4`.
        seed (int, optional): 
            Random seed for color mapping. Defaults to `42`.
        legend_font_size (int, optional): 
            Font size for legend labels. Defaults to `5`.
        rasterized (bool, optional): 
            Whether to rasterize the scatter plots to reduce file size. Defaults to `False`.
        save_dir (str, optional): 
            Directory where plots will be saved. Defaults to `'.'`.
        dpi (int, optional): 
            Image resolution in dots per inch. Defaults to `300`.
        save_format (str, optional): 
            Image format (`'png'`, `'pdf'`, `'svg'`, etc.). Defaults to `'png'`.
        file_suffix (str, optional): 
            Suffix to append to saved file names. Defaults to `'3d_plot'`.
        elev (float, optional): 
            Elevation angle for 3D view. Defaults to `30`.
        azim (float, optional): 
            Azimuth angle for 3D view. Defaults to `45`.
        zoom_factor (float, optional): 
            Zoom factor to adjust the scale of the plot. Defaults to `0.0`.
        **kwargs: 
            Additional parameters forwarded to `plot_embedding_3d_matplotlib` for customization.

    Returns:
        None: 
            Saves one figure per (basis_type, color_by) combination.

    Raises:
        ValueError: 
            If a specified `basis_type` is not found
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
                                      legend_itemsize=100, font_size=16, dpi=100, show_title=False, show_legend=False, 
                                      show_axes=False, show_background=False, elev=30, azim=0, zoom=1.5, seed=42):
    """
    Generates a rotating 3D embedding animation and saves it as an MP4 video.

    This function visualizes a 3D embedding (e.g., UMAP, PCA) with an animated rotation 
    and saves it as an MP4 video. The colors can be mapped to different cell metadata.
    By default, it shows only the point cloud for a clean visualization.

    Args:
        adata (AnnData): 
            Single-cell AnnData object containing embeddings and metadata.
        embedding_key (str, optional): 
            Key in `adata.obsm` where the 3D embedding is stored. Defaults to `'encoded_UMAP'`.
        color_by (str, optional): 
            Column in `adata.obs` to color points by. Defaults to `'batch'`.
        save_path (str, optional): 
            File path to save the MP4 video. Defaults to `'rotation.mp4'`.
        pal (dict, optional): 
            Color palette mapping categorical values to colors. Defaults to None.
        point_size (int, optional): 
            Size of the scatter plot points. Defaults to `3`.
        opacity (float, optional): 
            Opacity of the scatter plot points. Defaults to `0.7`.
        width (int, optional): 
            Width of the output video in pixels. Defaults to `800`.
        height (int, optional): 
            Height of the output video in pixels. Defaults to `1200`.
        rotation_duration (int, optional): 
            Duration of the rotation animation in seconds. Defaults to `10`.
        num_steps (int, optional): 
            Number of frames used for the rotation. Higher values result in a smoother animation. Defaults to `60`.
        legend_itemsize (int, optional): 
            Size of legend markers for categorical color mappings. Defaults to `100`.
        font_size (int, optional): 
            Font size for legends and labels. Defaults to `16`.
        dpi (int, optional):
            Resolution in dots per inch for the output images. Higher values result in higher resolution. Defaults to `100`.
        show_title (bool, optional):
            Whether to show the plot title. Defaults to `False`.
        show_legend (bool, optional):
            Whether to show the color legend or colorbar. Defaults to `False`.
        show_axes (bool, optional):
            Whether to show the axes and grid. Defaults to `False`.
        show_background (bool, optional):
            Whether to show the background. Defaults to `False`.
        elev (float, optional):
            Elevation angle in degrees (vertical rotation). Defaults to `30`.
        azim (float, optional):
            Azimuth angle in degrees (horizontal rotation). Defaults to `0`.
        zoom (float, optional):
            Zoom factor. Lower values zoom in, higher values zoom out. Defaults to `1.5`.
        seed (int, optional): 
            Random seed for color mapping. Defaults to `42`.

    Returns:
        None: 
            Saves the rotating animation as an MP4 video.

    Raises:
        KeyError: 
            If `embedding_key` is not found in `adata.obsm` or `color_by` is not in `adata.obs`.
        ValueError: 
            If the specified embedding has fewer than 3 dimensions.

    Example:
        ```python
        plot_rotating_embedding_3d_to_mp4(
            adata,
            embedding_key='X_umap',
            color_by='cell_type',
            save_path='3D_rotation.mp4',
            rotation_duration=15,
            num_steps=90,
            dpi=300,
            show_legend=True,
            elev=20,
            azim=45
        )
        ```
    """
    import numpy as np
    import plotly.graph_objs as go
    import plotly.express as px
    import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
    import pandas as pd
    import matplotlib.colors as mcolors
    import os
    import math
    
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
    
    # Set the title based on show_title parameter
    title = f'3D Embedding colored by {color_by}' if show_title else None
    
    # Determine if we're dealing with numeric data
    is_numeric = pd.api.types.is_numeric_dtype(data_col)
        
    # Plot based on data type: numeric or categorical
    if is_numeric:
        colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 256)]
        colorscale = [[i / (len(colors) - 1), color] for i, color in enumerate(colors)]
        fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=color_by,
                            title=title,
                            labels={'color': color_by}, opacity=opacity,
                            color_continuous_scale=colorscale)
    else:
        fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=color_by,
                            title=title,
                            labels={'color': color_by}, opacity=opacity,
                            color_discrete_map=palette)
        
        # Increase size of the points in the legend if legend is shown
        if show_legend:
            fig.update_layout(
                legend=dict(
                    font=dict(size=font_size),  # Increase legend font size
                    itemsizing='constant',  # Make legend items the same size
                    itemwidth=max(legend_itemsize, 30)  # Increase legend item width
                )
            )

    fig.update_traces(marker=dict(size=point_size), selector=dict(mode='markers'))
    
    # Configure layout based on parameters
    layout_updates = dict(width=width, height=height)
    
    # Hide legend/colorbar if not show_legend
    if not show_legend:
        if is_numeric:
            # For numeric data, hide the colorbar
            fig.update_layout(coloraxis_showscale=False)
            for trace in fig.data:
                if hasattr(trace, 'colorbar'):
                    trace.colorbar.showticklabels = False
                    trace.colorbar.thickness = 0
                    trace.colorbar.len = 0
        else:
            # For categorical data, hide the legend
            layout_updates['showlegend'] = False
    
    # Configure scene for axes and background
    scene_updates = {}
    
    if not show_axes:
        scene_updates.update({
            'xaxis': {'visible': False, 'showticklabels': False, 'showgrid': False, 'zeroline': False},
            'yaxis': {'visible': False, 'showticklabels': False, 'showgrid': False, 'zeroline': False},
            'zaxis': {'visible': False, 'showticklabels': False, 'showgrid': False, 'zeroline': False}
        })
    
    if not show_background:
        scene_updates['bgcolor'] = 'rgba(0,0,0,0)'  # Transparent background
    
    if scene_updates:
        layout_updates['scene'] = scene_updates
    
    # Apply all layout updates
    fig.update_layout(**layout_updates)
    
    # For a completely clean look without title
    if not show_title:
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    
    # Convert elevation and azimuth (degrees) to coordinates on a sphere for initial view
    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)
    x = zoom * math.cos(elev_rad) * math.cos(azim_rad)
    y = zoom * math.cos(elev_rad) * math.sin(azim_rad)
    z = zoom * math.sin(elev_rad)
    
    # Create rotation steps by defining camera positions
    def generate_camera_angles(num_steps, initial_x=x, initial_y=y, initial_z=z):
        # Start from the initial viewpoint and rotate around the z-axis
        initial_radius_xy = math.sqrt(initial_x**2 + initial_y**2)
        initial_theta = math.atan2(initial_y, initial_x)
        
        return [
            dict(
                eye=dict(
                    x=initial_radius_xy * math.cos(initial_theta + 2 * math.pi * t / num_steps),
                    y=initial_radius_xy * math.sin(initial_theta + 2 * math.pi * t / num_steps),
                    z=initial_z
                )
            )
            for t in range(num_steps)
        ]

    # Generate camera angles for the rotation
    camera_angles = generate_camera_angles(num_steps)
    print(f"number of frames: {len(camera_angles)}")
    
    # Save the frames as images with specified DPI
    frame_files = []
    for i, camera_angle in enumerate(camera_angles):
        fig.update_layout(scene_camera=camera_angle)
        frame_file = f"frame_{i:04d}.png"
        # Pass the DPI parameter to write_image
        fig.write_image(frame_file, scale=dpi/100)  # Convert DPI to scale factor
        frame_files.append(frame_file)

    # Create the video using MoviePy's ImageSequenceClip
    fps = num_steps / rotation_duration
    video = ImageSequenceClip.ImageSequenceClip(frame_files, fps=fps)
    video.write_videofile(save_path)

    # Clean up the temporary image files
    for frame_file in frame_files:
        os.remove(frame_file)

    print(f"Rotation video saved to {save_path}")
    
