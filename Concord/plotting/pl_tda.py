# Code to visualize persistent homology of data
import numpy as np
import matplotlib.pyplot as plt

def plot_persistence_diagram(diagram, homology_dimensions=None, ax=None, show=True,
                            legend=True, legend_loc='lower right', label_axes=True, colormap='tab10',
                            marker_size=20, diagonal=True, title=None,
                            xlim=None, ylim=None):
    diagram = diagram[0] # This is due to how giotto-tda returns the diagram
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
    # Set legend position, and make legend points (not font) larger
    if legend:
        ax.legend(loc=legend_loc, markerscale=3, handletextpad=0.2)
        

    if show:
        plt.show()

    return ax


def plot_betti_curves(diagram, nbins=100, homology_dimensions=[0,1,2], title="Betti curves", ymax=10, ax=None, show=True, legend=True, legend_loc='upper right'):
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
    if legend:
        ax.legend(loc=legend_loc)

    if show:
        plt.show()
    
    return ax


def plot_betti_statistic(betti_stats_pivot, statistic, dimension, color='skyblue', figsize=(6,4), dpi=300, save_path=None):
    """
    Plots the specified Betti curve statistic across methods for a given dimension.

    Parameters:
    - betti_stats_pivot: DataFrame containing Betti curve statistics (with MultiIndex columns).
    - statistic: String specifying the statistic to plot ('Entropy', 'Mean', 'Median', 'Mode', 'Variance').
    - dimension: String specifying the dimension (e.g., 'Dim 0', 'Dim 1', 'Dim 2').

    Returns:
    - None
    """
    import pandas as pd
    # Ensure the statistic name is capitalized to match the DataFrame columns
    statistic = statistic.capitalize()

    # Access the specific column in the MultiIndex DataFrame
    try:
        # If columns have MultiIndex
        if isinstance(betti_stats_pivot.columns, pd.MultiIndex):
            data_series = betti_stats_pivot[(dimension, statistic)]
        else:
            # If columns are flattened
            column_name = f"{dimension}_{statistic}"
            data_series = betti_stats_pivot[column_name]
    except KeyError:
        print(f"Statistic '{statistic}' for dimension '{dimension}' not found in the DataFrame.")
        return

    # Plot the data
    plt.figure(figsize=figsize, dpi=dpi)
    data_series.plot(kind='bar', color=color)

    plt.ylabel(statistic)
    plt.title(f"{statistic} for {dimension} Across Methods")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()




def plot_betti_distance(distance_metrics_df, metric, color='teal', log_y = False, figsize=(6, 4), dpi=300, save_path=None):
    """
    Plots the specified distance metric across methods.

    Parameters:
    - distance_metrics_df: DataFrame containing distance metrics for each method.
    - metric: String specifying the distance metric to plot ('L1 Distance', 'L2 Distance', 'Total Relative Error').
    - color: Color of the bars in the plot (default: 'teal').
    - figsize: Tuple specifying the figure size (default: (6, 4)).
    - dpi: Resolution of the figure in dots per inch (default: 300).
    - save_path: Path to save the plot as an image file (default: None, meaning the plot will not be saved).

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    # Ensure the metric is correctly capitalized to match the DataFrame columns
    metric = metric.title()

    # Check if the specified metric exists in the DataFrame
    if metric not in distance_metrics_df.columns:
        print(f"Metric '{metric}' not found in the DataFrame.")
        return

    # Plot the data
    plt.figure(figsize=figsize, dpi=dpi)
    distance_metrics_df[metric].plot(kind='bar', color=color)

    if log_y:
        plt.yscale('log')
        plt.ylabel(f'Log {metric}')
    else:
        plt.ylabel(metric)
    plt.title(f"{metric} Across Methods")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()


