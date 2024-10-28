
# Code to compute and visualize persistent homology of data

import numpy as np
import matplotlib.pyplot as plt

def compute_persistent_homology(adata, key='X_pca', homology_dimensions=[0,1,2]):
    from gtda.homology import VietorisRipsPersistence
    data = adata.obsm[key][None, :, :]
    VR = VietorisRipsPersistence(homology_dimensions=homology_dimensions)  # Parameter explained in the text
    diagrams = VR.fit_transform(data)
    return diagrams

def plot_persistence_diagram(diagram, homology_dimensions=None, ax=None, show=True,
                            legend=True, label_axes=True, colormap='tab10',
                            marker_size=20, diagonal=True, title=None,
                            xlim=None, ylim=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if homology_dimensions is None:
        homology_dimensions = np.unique(diagram[:, 2])

    # Prepare colormap
    cmap = plt.get_cmap(colormap)
    colors = cmap.colors if hasattr(cmap, 'colors') else [cmap(i) for i in range(cmap.N)]
    color_dict = {dim: colors[i % len(colors)] for i, dim in enumerate(homology_dimensions)}

    # Plot points for each homology dimension
    for dim in homology_dimensions:
        idx = (diagram[:, 2] == dim)
        births = diagram[idx, 0]
        deaths = diagram[idx, 1]

        # Handle infinite deaths
        finite_mask = np.isfinite(deaths)
        infinite_mask = ~finite_mask

        # Plot finite points
        ax.scatter(births[finite_mask], deaths[finite_mask],
                   label=f'H{int(dim)}', s=marker_size, color=color_dict[dim])

        # Plot points at infinity (if any)
        if np.any(infinite_mask):
            max_finite = np.max(deaths[finite_mask]) if np.any(finite_mask) else np.max(births)
            infinite_death = max_finite + 0.1 * (max_finite - np.min(births))
            ax.scatter(births[infinite_mask], [infinite_death] * np.sum(infinite_mask),
                       marker='^', s=marker_size, color=color_dict[dim])

            # Adjust y-axis limit to accommodate infinity symbol
            if ylim is None:
                ax.set_ylim(bottom=ax.get_ylim()[0], top=infinite_death + 0.1 * infinite_death)
            # Add infinity symbol as a custom legend entry
            ax.scatter([], [], marker='^', label='Infinity', color='black')

    # Draw diagonal line
    if diagonal:
        limits = [
            np.min(np.concatenate([diagram[:, 0], diagram[:, 1]])),
            np.max(np.concatenate([diagram[:, 0], diagram[:, 1]]))
        ]
        ax.plot(limits, limits, 'k--', linewidth=1)

    if legend:
        ax.legend()

    if label_axes:
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')

    if title is not None:
        ax.set_title(title)

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Customize font sizes
    ax.tick_params(axis='both', which='major')
    ax.set_title(title)

    if show:
        plt.show()

    return ax


def plot_betti_curves(diagram, nbins=100, homology_dimensions=[0,1,2], title="Betti curves", ymax=10, ax=None, show=True):
    from gtda.diagrams import BettiCurve
    betti_curve = BettiCurve(n_bins=nbins)
    betti_curves = betti_curve.fit_transform(diagram)
    filtration_values = betti_curve.samplings_

    # Plot Betti curves for Betti-0, Betti-1, and Betti-2
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    for dim in homology_dimensions:
        ax.plot(filtration_values[dim], betti_curves[0][dim, :], label=f'Betti-{dim}')

    ax.set_xlabel('Filtration Parameter')
    ax.set_ylabel('Betti Numbers')
    ax.set_title(title)
    ax.set_ylim(0, ymax)
    ax.legend()

    if show:
        plt.show()
    
    return ax


def plot_mapper_graph(adata, key='X_pca', filter_func='projection', input_dim=3, layout_dim=2, 
                      cover_n_intervals=20, cover_overlap_frac=0.5, cluster_eps=0.5, 
                      color_by=None, cmap='viridis', node_scale=30, save_path=None, verbose=False):
    from gtda.mapper import CubicalCover, make_mapper_pipeline, Projection, plot_static_mapper_graph
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN

    data = adata.obsm[key]
    # Create a pipeline with a projection and a mapper
    if filter_func == 'projection':
        filter_func = Projection(columns=list(range(input_dim)))
    elif filter_func == 'PCA':
        filter_func = PCA(n_components=input_dim)
    else:
        raise ValueError("filter_func must be 'projection' or 'PCA'")
    
    cover =  CubicalCover(n_intervals=cover_n_intervals, overlap_frac=cover_overlap_frac)
    clusterer = DBSCAN(eps=cluster_eps)


    pipeline = make_mapper_pipeline(filter_func=filter_func, cover=cover, clusterer=clusterer, verbose=verbose)
    # Transform the data
    # Color nodes
    if color_by is None:
        df_color = data
    elif isinstance(color_by, str):
        if color_by in adata.obs.keys():
            df_color = adata.obs[color_by]
        else:
            raise ValueError(f"Key '{color_by}' not found in adata.obs")
    else:
        raise ValueError("color_data must be a string")
    
    plotly_params = {"node_trace": {"marker_colorscale": cmap}}

    fig = plot_static_mapper_graph(pipeline, data, layout_dim=layout_dim, node_scale=node_scale, color_data=df_color, plotly_params=plotly_params)

    if save_path:
        fig.write_html(save_path)

    fig.show()


