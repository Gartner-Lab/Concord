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


def plot_persistence_diagrams(diagrams, marker_size=4, n_cols=3, dpi=300, base_size=(3,3), save_path=None):
    # Plot the persistence diagrams into a single figure
    import matplotlib.pyplot as plt
    combined_keys = list(diagrams.keys())
    n_plots = len(combined_keys)
    n_rows = int(np.ceil(n_plots / n_cols))
    base_height = base_size[1]
    base_width = base_size[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(base_width * n_cols, base_height * n_rows), dpi=dpi)
    for i, key in enumerate(combined_keys):
        # avoid plotting empty subplots
        if i >= n_plots:
            break
        ax = axes[i // n_cols, i % n_cols]
        plot_persistence_diagram(diagrams[key], ax=ax, marker_size=marker_size, title=key, show=False)

    # Save the plot
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_betti_curve(diagram, nbins=100, homology_dimensions=[0,1,2], title="Betti curves", ymax=10, ax=None, show=True, legend=True, legend_loc='upper right'):
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


def plot_betti_curves(diagrams, nbins=100, ymax=8, n_cols=3, base_size=(3,3), dpi=300, legend=True, save_path=None):
    # Plot the betti curves into a single figure
    import matplotlib.pyplot as plt
    combined_keys = list(diagrams.keys())
    n_plots = len(combined_keys)
    n_rows = int(np.ceil(n_plots / n_cols))
    base_height = base_size[1]
    base_width = base_size[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(base_width * n_cols, base_height * n_rows), dpi=dpi)
    for i, key in enumerate(combined_keys):
        # avoid plotting empty subplots
        if i >= n_plots:
            break
        ax = axes[i // n_cols, i % n_cols]
        plot_betti_curve(diagrams[key], nbins=nbins, ymax=ymax, ax=ax, title=key, show=False, legend=legend)

    # Save the plot
    fig.tight_layout()

    if save_path:   
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()



def plot_betti_statistic(
    betti_stats_pivot, statistic='Entropy', dimension=None, log_y=False, bar_width=0.2, 
    pal='tab20', figsize=(7, 4), dpi=300, save_path=None,
    xlabel_fontsize=8, ylabel_fontsize=8, tick_fontsize=7, 
    title_fontsize=9, legend_fontsize=8
):
    """
    Plots a grouped bar plot for a specified statistic across dimensions for each method.
    Allows plotting all dimensions or a single specified dimension.

    Parameters:
    - betti_stats_pivot: DataFrame with multi-index columns (Dimension, Statistic).
    - statistic: The name of the statistic to plot (e.g., 'Entropy', 'Variance').
    - dimension: Specific dimension to plot (e.g., 'Dim 0') or None to plot all dimensions.
    - log_y: Boolean, whether to use a logarithmic scale for the y-axis.
    - bar_width: Width of each bar in the grouped bar plot (default is 0.2).
    - pal: Color palette for the bars (default is 'tab20').
    - figsize: Size of the figure (default is (10, 6)).
    - dpi: Resolution of the figure (default is 300).
    - save_path: Path to save the figure (default is None).
    - xlabel_fontsize, ylabel_fontsize, tick_fontsize, title_fontsize, legend_fontsize: Font sizes.
    
    Returns:
    - None
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter to the specified dimension if provided
    if dimension is not None:
        dimension = f"Dim {dimension}" if isinstance(dimension, int) else dimension
        stat_columns = betti_stats_pivot.loc[:, pd.IndexSlice[dimension, statistic]].to_frame()
        stat_columns.columns = [dimension]  # Rename the column to match the dimension
    else:
        stat_columns = betti_stats_pivot.loc[:, pd.IndexSlice[:, statistic]]
        stat_columns.columns = [f"Dim {i}" for i in range(stat_columns.shape[1])]

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Define the number of methods and dimensions to plot
    n_methods = len(stat_columns.index)
    n_dimensions = len(stat_columns.columns)
    indices = np.arange(n_methods)
    
    # Generate color palette
    colors = sns.color_palette(pal, n_dimensions) if isinstance(pal, str) else pal

    # Plot each dimension's data as a separate set of bars
    for i, (dim, color) in enumerate(zip(stat_columns.columns, colors)):
        ax.bar(
            indices + i * bar_width,
            stat_columns[dim],
            width=bar_width,
            color=color,
            label=dim
        )
    
    # Set labels and titles with custom font sizes
    ax.set_xticks(indices + bar_width * (n_dimensions - 1) / 2)
    ax.set_xticklabels(stat_columns.index, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.set_ylabel(statistic, fontsize=ylabel_fontsize)
    ax.set_xlabel("Method", fontsize=xlabel_fontsize)
    if log_y:
        ax.set_yscale('log')

    # Title with custom font size
    title_dimension = f" for {dimension}" if dimension else " across Dimensions"
    ax.set_title(f'{statistic}{title_dimension} for Each Method', fontsize=title_fontsize)

    # Legend with custom font size
    ax.legend(
        title=None, 
        loc='center left', 
        bbox_to_anchor=(1, 0.5),
        markerscale=1.5,
        handletextpad=0.2,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize
    )
    
    # Adjust layout and show plot
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)

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


