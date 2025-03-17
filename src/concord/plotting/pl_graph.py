from .. import logger
from .palettes import get_color_mapping
import pandas as pd
import warnings

# Not used, scanpy has a faster version
def visualize_knn_graph(
    adata, 
    basis='X_pca', 
    dim=2,
    color_by=None, 
    cmap='viridis',
    k=10, 
    layout='spring', 
    node_size=10,
    draw_edges=False, 
    edge_color='gray',
    edge_width=0.1,
    plotly=False,
    ax=None,
    seed=42
):
    """
    Visualizes a k-NN graph from an AnnData object using matplotlib or Plotly.

    Parameters:
    - adata: AnnData object
        The data object containing the data and metadata.
    - basis: str, default='X_pca'
        The basis in adata.obsm, adata.X, or adata.layers for graph computation.
    - color_by: str, default='depth'
        Key in adata.obs to color nodes.
    - k: int, default=5
        Number of neighbors for k-NN graph.
    - layout: str, default='spring'
        Layout algorithm for visualization ('spring', 'spectral', or 'circular').
    - draw_edges: bool, default=True
        Whether to draw edges in the graph.
    - plotly_3d: bool, default=False
        If True, use a 3D Plotly visualization, otherwise use matplotlib.
    - ax: matplotlib.axes._subplots.AxesSubplot, default=None
        Matplotlib axis object to plot on. If None, a new figure is created.
    """
    import networkx as nx
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    # Get data based on the chosen basis
    if basis in adata.obsm:
        data = adata.obsm[basis]
    elif basis in adata.layers:
        data = adata.layers[basis]
    elif basis == 'X':
        data = adata.X
    else:
        raise ValueError(f"Invalid basis '{basis}', choose a basis from adata.obsm, adata.X, or adata.layers")

    # Build k-NN graph
    logger.info(f"Building {k}-NN graph using {basis} basis")
    nn = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nn.kneighbors(data)
    G = nx.Graph()
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:
                G.add_edge(i, neighbor)

    logger.info(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Set node colors based on `color_by`
    if color_by is not None:
        if color_by in adata.obs:
            data_col, cmap, palette = get_color_mapping(adata, color_by, {color_by: cmap})
            
            # Handle color mapping for numeric and categorical data
            if pd.api.types.is_numeric_dtype(data_col):
                node_colors = data_col
                is_numeric = True
            else:
                data_col = data_col.astype('category')
                node_colors = data_col.map({cat: palette[i] for i, cat in enumerate(data_col.cat.categories)})
                cmap = None
                is_numeric = False
        else:
            raise ValueError(f"Key '{color_by}' not found in adata.obs")
    else:
        node_colors = np.zeros(adata.shape[0])
        is_numeric = True

    # Assign color as node attribute
    for i, color in enumerate(node_colors):
        G.nodes[i]['color'] = color

    node_color_values = np.array([G.nodes[v]['color'] for v in G.nodes])

    # Choose layout
    logger.info(f"Computing {layout} layout for visualization")
    if layout == 'spring':
        pos = nx.spring_layout(G, dim=dim, seed=seed)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G, dim=dim)
    elif layout == 'circular':
        pos = nx.circular_layout(G, dim=dim)
    elif layout == 'forceatlas2':
        pos = nx.forceatlas2_layout(G, dim=dim, seed=seed)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, dim=dim)
    else:
        raise ValueError(f"Unsupported layout '{layout}'")

    logger.info(f"Plotting {dim}D visualization")
    # 2D Visualization with Matplotlib
    if dim == 2:
        node_positions = np.array([pos[v] for v in G.nodes()])
        
        # Use provided ax if available, otherwise create a new figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw nodes
        sx = ax.scatter(
            node_positions[:, 0], node_positions[:, 1],
            c=node_color_values, cmap=cmap, s=node_size
        )
        
        # Draw edges if enabled
        if draw_edges:
            for i, j in G.edges():
                ax.plot(
                    [pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                    color=edge_color, linewidth=edge_width
                )

        if is_numeric:
            plt.colorbar(sx, ax=ax, label=color_by)
        else:
            ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10, label=cat) for i, cat in enumerate(data_col.cat.categories)], title=color_by)

        ax.set_title(f"{k}-Nearest Neighbors Graph ({layout} layout)")
        ax.axis('off')

        # Show only if no ax was supplied
        if ax is None:
            plt.show()
    
    # 3D Visualization with Plotly
    elif dim == 3:
        if not plotly:
            node_positions = np.array([pos[v] for v in G.nodes()])
    
            # Set up the 3D plot
            if ax is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
            
            # Draw nodes
            sc = ax.scatter(
                node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                c=node_color_values, 
                cmap=cmap, 
                s=node_size
            )
            
            # Draw edges if enabled
            if draw_edges:
                for i, j in G.edges():
                    ax.plot(
                        [pos[i][0], pos[j][0]], 
                        [pos[i][1], pos[j][1]], 
                        [pos[i][2], pos[j][2]], 
                        color=edge_color, linewidth=edge_width
                    )
            
            # Add color bar
            if ax is None:
                cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
                cbar.set_label(color_by)
            
            # Set plot title and turn off the grid for a cleaner look
            ax.set_title(f"{k}-Nearest Neighbors Graph ({layout} layout)")
            ax.grid(False)
            
            if ax is None:
                plt.show()

        else:
            # 3D Plotly visualization remains unchanged since Plotly handles axes differently
            node_xyz = np.array([pos[v] for v in G.nodes()])
            edge_x, edge_y, edge_z = [], [], []
            if draw_edges:
                for i, j in G.edges():
                    edge_x += [pos[i][0], pos[j][0], None]
                    edge_y += [pos[i][1], pos[j][1], None]
                    edge_z += [pos[i][2], pos[j][2], None]
            
            # Create node scatter plot
            node_trace = go.Scatter3d(
                x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
                mode='markers',
                marker=dict(
                    size=node_size,
                    color=node_color_values,
                    colorscale=cmap,
                    colorbar=dict(title=color_by),
                ),
                text=[f"Node {v}" for v in G.nodes()],
                hoverinfo='text'
            )
            
            # Create edge scatter plot
            if draw_edges:
                edge_trace = go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color=edge_color, width=edge_width),
                    hoverinfo='none'
                )
                fig = go.Figure(data=[edge_trace, node_trace])
            else:
                fig = go.Figure(data=[node_trace])

            fig.update_layout(
                title=f"{k}-Nearest Neighbors Graph ({layout} layout)",
                showlegend=False,
                scene=dict(
                    xaxis=dict(showbackground=False),
                    yaxis=dict(showbackground=False),
                    zaxis=dict(showbackground=False)
                )
            )
            fig.show()
    else:
        raise ValueError("Only 2D and 3D visualizations are supported")






def plot_graph(adata, basis, 
               k=15, layout='kk', seed=42,
               color_by=None, edges=False, edges_width=0.1, edges_color='grey',
                   pal=None, highlight_indices=None,
                   highlight_size=20, draw_path=False, alpha=0.9, text_alpha=0.5,
                   figsize=(9, 3), dpi=300, ncols=1,
                   font_size=8, point_size=10, legend_loc='on data', save_path=None):
    import scanpy as sc
    import matplotlib.pyplot as plt
    import numpy as np

    warnings.filterwarnings('ignore')

    if color_by is None or len(color_by) == 0:
        color_by = [None]  # Use a single plot without coloring

    nrows = int(np.ceil(len(color_by) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
    axs = np.atleast_2d(axs).flatten()  # Ensure axs is a 1D array for easy iteration

    # Compute knn graph and set graph layout
    sc.pp.neighbors(adata, n_neighbors=k, use_rep=basis, random_state=seed)
    sc.tl.draw_graph(adata, layout=layout, random_state=seed)
    
    if not isinstance(pal, dict):
        pal = {col: pal for col in color_by}

    for col, ax in zip(color_by, axs):
        data_col, cmap, palette = get_color_mapping(adata, col, pal)

        if col is None:
            ax = sc.pl.draw_graph(adata, color=col, ax=ax, show=False,
                                 legend_loc='right margin', legend_fontsize=font_size,
                                 size=point_size, alpha=alpha, edges=edges, edges_width=edges_width, edges_color=edges_color)
        elif pd.api.types.is_numeric_dtype(data_col):
            sc.pl.draw_graph(adata, color=col, ax=ax, show=False,
                            legend_loc='right margin', legend_fontsize=font_size,
                            size=point_size, alpha=alpha, cmap=cmap, edges=edges, edges_width=edges_width, edges_color=edges_color)
        else:
            sc.pl.draw_graph(adata, color=col, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=font_size,
                            size=point_size, alpha=alpha, palette=palette, edges=edges, edges_width=edges_width, edges_color=edges_color)

        if legend_loc == 'on data':
            for text in ax.texts:
                text.set_alpha(text_alpha)

        # Highlight selected points, check if needs fixing
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

        ax.set_title(ax.get_title(), fontsize=font_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)

        if hasattr(ax, 'collections') and len(ax.collections) > 0:
            cbar = ax.collections[-1].colorbar
            if cbar is not None:
                cbar.ax.tick_params(labelsize=font_size)

    for ax in axs[len(color_by):]:
        ax.axis('off')

    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)


# 
def plot_paga(adata, basis, 
            k=15, groups='leiden', resolution=1.0, 
            threshold=0.1,
            layout='kk',
            seed=42,
            color_by=None, 
            node_size_scale=1,
            edge_width_scale=1,
            pal=None, 
            figsize=(9, 3), dpi=300, ncols=1,
            font_size=8, save_path=None):
    import scanpy as sc
    import matplotlib.pyplot as plt
    import numpy as np

    warnings.filterwarnings('ignore')

    if color_by is None or len(color_by) == 0:
        color_by = [None]  # Use a single plot without coloring

    nrows = int(np.ceil(len(color_by) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, constrained_layout=True)
    axs = np.atleast_2d(axs).flatten()  # Ensure axs is a 1D array for easy iteration

    # Compute knn graph and set graph layout
    sc.pp.neighbors(adata, n_neighbors=k, use_rep=basis, random_state=seed)
    
    if groups =='leiden':
        sc.tl.leiden(adata, resolution=resolution)
    elif groups == 'louvain':
        sc.tl.louvain(adata, resolution=resolution)
    else:
        if groups not in adata.obs:
            raise ValueError(f"Group key '{groups}' not found in adata.obs")
    
    sc.tl.paga(adata, groups=groups)
    
    if not isinstance(pal, dict):
        pal = {col: pal for col in color_by}

    for col, ax in zip(color_by, axs):
        data_col, cmap, _ = get_color_mapping(adata, col, pal)

        if col is None:
            sc.pl.paga(adata, threshold=threshold, color=col, ax=ax, show=False,
                            layout=layout, fontsize=font_size,
                            cmap=cmap, node_size_scale=node_size_scale, edge_width_scale=edge_width_scale)
        elif pd.api.types.is_numeric_dtype(data_col):
            sc.pl.paga(adata, threshold=threshold, color=col, ax=ax, show=False,
                        layout=layout, fontsize=font_size,
                        cmap=cmap, node_size_scale=node_size_scale, edge_width_scale=edge_width_scale, colorbar=False)
        else:
            sc.tl.paga(adata, groups=groups) # Strange that this needs to be run right before otherwise it doesn't work
            sc.pl.paga(adata, threshold=threshold, color=col, ax=ax, show=False,
                            layout=layout, fontsize=font_size,
                            cmap=cmap, node_size_scale=node_size_scale, edge_width_scale=edge_width_scale)

        ax.set_title(ax.get_title(), fontsize=font_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)


    for ax in axs[len(color_by):]:
        ax.axis('off')

    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)





def compute_paga_layout(adata, groupby_key, weight_threshold=0.3, spring_k=None, seed=42):
    """
    Precompute the PAGA graph layout positions with filtered edges.

    Parameters:
    - adata: AnnData object
    - groupby_key: str, column in adata.obs used to group cells (e.g., 'leiden_Concord')
    - weight_threshold: float, minimum weight for edges to be included in the graph
    - spring_k: float, spring constant for layout positioning
    - seed: int, random seed for reproducibility

    Returns:
    - filtered_graph: NetworkX graph object
    - pos: Dictionary of positions for the nodes
    """
    import networkx as nx

    # Extract the PAGA adjacency matrix
    paga_connectivities = adata.uns['paga']['connectivities']

    # Convert to list of edges with weights
    edges = [
        (i, j, w)
        for i, j, w in zip(*paga_connectivities.nonzero(), paga_connectivities.data)
    ]

    # Filter edges by weight threshold
    filtered_edges = [(i, j, w) for i, j, w in edges if w >= weight_threshold]

    # Build the filtered graph
    filtered_graph = nx.Graph()
    filtered_graph.add_weighted_edges_from(filtered_edges)

    # Relabel nodes to use group labels
    label_mapping = {i: str(label) for i, label in enumerate(adata.obs[groupby_key].cat.categories)}
    filtered_graph = nx.relabel_nodes(filtered_graph, label_mapping)

    # Precompute layout positions
    pos = nx.spring_layout(filtered_graph, weight='weight', k=spring_k, seed=seed)
    return filtered_graph, pos


def plot_paga_custom(adata, filtered_graph, pos, meta_attribute_key, groupby_key, node_size=500, with_labels=True, figsize=(10, 10), pal=None, save_path=None):
    """
    Plot a PAGA graph using a shared layout and colored based on a meta attribute.

    Parameters:
    - adata: AnnData object
    - filtered_graph: Precomputed NetworkX graph
    - pos: Dictionary of shared node positions
    - meta_attribute_key: str, column in adata.obs for node coloring
    - groupby_key: str, column in adata.obs used to group cells
    - node_size: int, size of the nodes in the plot
    - with_labels: bool, whether to display node labels
    - figsize: tuple, size of the figure
    - save_path: str, optional path to save the figure
    """    
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, to_hex
    from . import get_color_mapping
    # Step 1: Assign the meta attribute as a node attribute
    meta_attribute = adata.obs.groupby(groupby_key)[meta_attribute_key].first()
    meta_attribute_dict = meta_attribute.to_dict()
    nx.set_node_attributes(filtered_graph, {node: meta_attribute_dict.get(node, 'Unknown') for node in filtered_graph.nodes}, name=meta_attribute_key)

    # Step 2: Handle color mapping
    if pd.api.types.is_numeric_dtype(adata.obs[meta_attribute_key]):
        # Continuous colormap for numeric values
        values = [meta_attribute_dict[node] for node in filtered_graph.nodes]
        norm = Normalize(vmin=min(values), vmax=max(values))
        data_col, cmap, palette = get_color_mapping(adata, meta_attribute_key, pal=pal)
        node_colors = [to_hex(cmap(norm(meta_attribute_dict[node]))) for node in filtered_graph.nodes]
    else:
        # Discrete colormap for categorical values
        unique_values = meta_attribute.unique()
        data_col, cmap, palette = get_color_mapping(adata, meta_attribute_key, pal=pal)
        node_colors = [palette[filtered_graph.nodes[node][meta_attribute_key]] for node in filtered_graph.nodes]

    # Step 3: Plot the graph
    plt.figure(figsize=figsize)

    # Node labels are the meta attributes
    node_labels = {node: filtered_graph.nodes[node][meta_attribute_key] for node in filtered_graph.nodes} if with_labels else None

    nx.draw(
        filtered_graph,
        pos,
        with_labels=with_labels,
        labels=node_labels,
        node_color=node_colors,
        edge_color='gray',
        node_size=node_size,
        font_size=10
    )

    plt.title(f"PAGA Graph Colored by '{meta_attribute_key}'")
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

