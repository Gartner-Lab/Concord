
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
                   font_size=8, point_size=10, path_width=1, legend_loc='on data', 
                   rasterized=True,
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
        data_col, cmap, palette = get_color_mapping(adata, col, pal)

        if col is None:
            sc.pl.embedding(adata, basis=basis, ax=ax, show=False,
                                 legend_loc='right margin', legend_fontsize=font_size,
                                 size=point_size, alpha=alpha)
            for collection in ax.collections:
                collection.set_color(default_color)
        elif pd.api.types.is_numeric_dtype(data_col):
            sc.pl.embedding(adata, basis=basis, color=col, ax=ax, show=False,
                            legend_loc='right margin', legend_fontsize=font_size,
                            size=point_size, alpha=alpha, cmap=cmap, colorbar_loc=colorbar_loc)
        else:
            sc.pl.embedding(adata, basis=basis, color=col, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=font_size,
                            size=point_size, alpha=alpha, palette=palette)

        if legend_loc == 'on data':
            for text in ax.texts:
                text.set_alpha(text_alpha)

        # Highlight selected points
        if highlight_indices is not None:
            highlight_data = adata[highlight_indices, :]
            if col is None:
                sc.pl.embedding(highlight_data, basis=basis, ax=ax, show=False,
                                legend_loc=None, legend_fontsize=font_size,
                                size=highlight_size, alpha=1.0)
                for collection in ax.collections:
                    collection.set_color(highlight_color)
            elif pd.api.types.is_numeric_dtype(data_col):
                sc.pl.embedding(highlight_data, basis=basis, color=col, ax=ax, show=False,
                                legend_loc=None, legend_fontsize=font_size,
                                size=highlight_size, alpha=1.0, cmap=cmap, colorbar_loc=None)
            else:
                sc.pl.embedding(highlight_data, basis=basis, color=col, ax=ax, show=False,
                                legend_loc=None, legend_fontsize=font_size,
                                size=highlight_size, alpha=1.0, palette=palette)

            if draw_path:
                embedding = adata.obsm[basis]
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                path_coords = embedding[highlight_indices, :]
                ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=path_width)  # Red line for the path

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


def plot_embedding_3d(adata, basis='encoded_UMAP', color_by='batch', pal=None,  
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
        # Retrieve color mapping using the new get_color_mapping method
        data_col, cmap, palette = get_color_mapping(adata, col, pal)
        # Plot based on data type: numeric or categorical
        if pd.api.types.is_numeric_dtype(data_col):
            colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 256)]
            colorscale = [[i / (len(colors) - 1), color] for i, color in enumerate(colors)]
            fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=col,
                                title=f'3D Embedding colored by {col}',
                                labels={'color': col}, opacity=opacity,
                                color_continuous_scale=colorscale)
        else:
            fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=col,
                                title=f'3D Embedding colored by {col}',
                                labels={'color': col}, opacity=opacity,
                                color_discrete_map=palette)

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





def plot_all_embeddings(
    adata,
    combined_keys,
    color_bys=['time', 'batch'],
    basis_types=['PAGA', 'KNN', 'PCA', 'UMAP'],
    pal={'time': 'viridis', 'batch': 'Set1'},
    k=15,
    edges_color='grey',
    edges_width=0.05,
    layout='kk',
    threshold=0.1,
    node_size_scale=0.1,
    edge_width_scale=0.1,
    font_size=7,
    point_size=2.5,
    alpha=0.8,
    figsize=(9, 0.9),
    ncols=11,
    seed=42,
    leiden_key='leiden',
    leiden_resolution=1.0,
    legend_loc = None,
    colorbar_loc = None,
    rasterized=True,
    save_dir='.',
    save_format='png',
    file_suffix='plot'
):
    
    nrows = int(np.ceil(len(combined_keys) / ncols))

    for basis_type in basis_types:
        print(f"Plotting {basis_type} embeddings")
        for color_by in color_bys:
            print(f"Coloring by {color_by}")
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=300, constrained_layout=True)
            axs = np.atleast_2d(axs).flatten()  # Ensure axs is a 1D array for easy iteration

            for key, ax in zip(combined_keys, axs):
                data_col, cmap, palette = get_color_mapping(adata, color_by, pal)
                basis = f'{key}_{basis_type}' if basis_type not in key else key

                if basis_type in ['PCA', 'UMAP']:
                    if pd.api.types.is_numeric_dtype(data_col):
                        sc.pl.embedding(
                            adata, basis=basis, color=color_by, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=font_size,
                            size=point_size, alpha=alpha, cmap=cmap, colorbar_loc=colorbar_loc
                        )
                    else:
                        sc.pl.embedding(
                            adata, basis=basis, color=color_by, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=font_size,
                            size=point_size, alpha=alpha, palette=palette
                        )

                elif basis_type == 'KNN':
                    sc.pp.neighbors(adata, n_neighbors=k, use_rep=key, random_state=seed)    
                    sc.tl.draw_graph(adata, layout=layout, random_state=seed)
                    if pd.api.types.is_numeric_dtype(data_col):
                        sc.pl.draw_graph(
                            adata, color=color_by, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=font_size,
                            size=point_size, alpha=alpha, cmap=cmap, edges=True,
                            edges_width=edges_width, edges_color=edges_color, colorbar_loc=colorbar_loc
                        )
                    else:
                        sc.pl.draw_graph(
                            adata, color=color_by, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=font_size,
                            size=point_size, alpha=alpha, palette=palette,
                            edges=True, edges_width=edges_width, edges_color=edges_color
                        )

                elif basis_type == 'PAGA':
                    sc.pp.neighbors(adata, n_neighbors=k, use_rep=key, random_state=seed) 
                    sc.tl.leiden(adata, key_added=leiden_key, resolution=leiden_resolution, random_state=seed)   
                    sc.tl.paga(adata, groups=leiden_key, use_rna_velocity=False)
                    if pd.api.types.is_numeric_dtype(data_col):
                        sc.pl.paga(
                            adata, threshold=threshold, color=color_by, ax=ax, show=False,
                            layout=layout, fontsize=2, cmap=cmap, node_size_scale=node_size_scale,
                            edge_width_scale=edge_width_scale, colorbar=False
                        )
                    else:
                        sc.pl.paga(
                            adata, threshold=threshold, color=color_by, ax=ax, show=False,
                            layout=layout, fontsize=2, cmap=cmap, node_size_scale=node_size_scale,
                            edge_width_scale=edge_width_scale
                        )

                if 'PCA' in key:
                    key = key.replace('PCA_', '')
                ax.set_title(key, fontsize=font_size)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticks([])
                ax.set_yticks([])

                if rasterized:
                    import matplotlib.collections as mcoll
                    for artist in ax.get_children():
                        if isinstance(artist, mcoll.PathCollection):
                            artist.set_rasterized(True)
                        if isinstance(artist, mcoll.LineCollection):  # Find the edges
                            artist.set_rasterized(True)

            # Hide any remaining empty axes
            for ax in axs[len(combined_keys):]:
                ax.set_visible(False)

            # Save the figure
            plt.savefig(f"{save_dir}/all_latent_{color_by}_{basis_type}_{file_suffix}.{save_format}", bbox_inches=None)
            plt.show()





def plot_rotating_embedding_3d_to_mp4(adata, embedding_key='encoded_UMAP', color_by='batch', save_path='rotation.mp4', pal=None,
                                      point_size=3, opacity=0.7, width=800, height=1200, rotation_duration=10, num_steps=60,
                                      legend_itemsize=100, font_size=16):
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
    data_col, cmap, palette = get_color_mapping(adata, color_by, pal)
        
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
