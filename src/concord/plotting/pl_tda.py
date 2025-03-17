# Code to visualize persistent homology of data
import numpy as np
import matplotlib.pyplot as plt

def plot_persistence_diagram(diagram, homology_dimensions=None, ax=None, show=True,
                            legend=True, legend_loc='lower right', label_axes=True, colormap='tab10',
                            marker_size=20, diagonal=True, title=None, fontsize=12, axis_ticks=True,
                            xlim=None, ylim=None, rasterized=True):
    """
    Plots a persistence diagram showing birth and death times of topological features.

    Args:
        diagram (array-like): Persistence diagram data (birth, death, homology dimension).
        homology_dimensions (list, optional): 
            Homology dimensions to plot (e.g., [0, 1, 2]). Defaults to all available.
        ax (matplotlib.axes.Axes, optional): 
            Matplotlib axis to plot on. If `None`, a new figure is created.
        show (bool, optional): 
            Whether to display the plot. Defaults to `True`.
        legend (bool, optional): 
            Whether to show a legend. Defaults to `True`.
        legend_loc (str, optional): 
            Location of the legend. Defaults to `'lower right'`.
        label_axes (bool, optional): 
            Whether to label the x- and y-axes. Defaults to `True`.
        colormap (str, optional): 
            Colormap for different homology dimensions. Defaults to `'tab10'`.
        marker_size (int, optional): 
            Size of markers for points. Defaults to `20`.
        diagonal (bool, optional): 
            Whether to plot the diagonal y = x reference line. Defaults to `True`.
        title (str, optional): 
            Title of the plot. Defaults to `None`.
        fontsize (int, optional): 
            Font size for labels and title. Defaults to `12`.
        axis_ticks (bool, optional): 
            Whether to display axis ticks. Defaults to `True`.
        xlim (tuple, optional): 
            Limits for the x-axis. Defaults to `None`.
        ylim (tuple, optional): 
            Limits for the y-axis. Defaults to `None`.
        rasterized (bool, optional): 
            Whether to rasterize the plot for performance. Defaults to `True`.

    Returns:
        matplotlib.axes.Axes: The axis object containing the persistence diagram.

    Example:
        ```python
        plot_persistence_diagram(diagram, homology_dimensions=[0, 1])
        ```
    """
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
                   label=f'H{int(dim)}', s=marker_size, color=color_dict[dim], rasterized=rasterized)

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
            ax.scatter([], [], marker='^', label='Infinity', color='black', rasterized=rasterized)

    # Draw diagonal line
    if diagonal:
        limits = [
            np.min(np.concatenate([diagram[:, 0], diagram[:, 1]])),
            np.max(np.concatenate([diagram[:, 0], diagram[:, 1]]))
        ]
        ax.plot(limits, limits, 'k--', linewidth=1)

    if label_axes:
        ax.set_xlabel('Birth', fontsize=fontsize-2)
        ax.set_ylabel('Death', fontsize=fontsize-2)

    if title is not None:
        ax.set_title(title)

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set axis ticks
    if not axis_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Customize font sizes
    ax.tick_params(axis='both', which='major')
    ax.set_title(title, fontsize=fontsize)
    # Set legend position, and make legend points (not font) larger
    if legend:
        ax.legend(loc=legend_loc, markerscale=3, handletextpad=0.2, fontsize=fontsize-1)
        
    if show:
        plt.show()

    return ax


def plot_persistence_diagrams(diagrams, marker_size=4, n_cols=3, dpi=300, base_size=(3,3), legend=True, legend_loc=None, rasterized=True, fontsize=12, save_path=None, **kwargs):
    """
    Plots multiple persistence diagrams in a grid layout.

    Args:
        diagrams (dict): 
            Dictionary where keys are dataset names and values are persistence diagrams.
        marker_size (int, optional): 
            Size of markers for points. Defaults to `4`.
        n_cols (int, optional): 
            Number of columns in the grid. Defaults to `3`.
        dpi (int, optional): 
            Resolution of the figure. Defaults to `300`.
        base_size (tuple, optional): 
            Base figure size for each subplot `(width, height)`. Defaults to `(3, 3)`.
        legend (bool, optional): 
            Whether to include legends. Defaults to `True`.
        legend_loc (str, optional): 
            Location of the legend. Defaults to `None`.
        rasterized (bool, optional): 
            Whether to rasterize the plots. Defaults to `True`.
        fontsize (int, optional): 
            Font size for labels and titles. Defaults to `12`.
        save_path (str, optional): 
            File path to save the figure. Defaults to `None`.

    Returns:
        None

    Example:
        ```python
        plot_persistence_diagrams(diagrams, n_cols=2, save_path="persistence_diagrams.png")
        ```
    """
    # Plot the persistence diagrams into a single figure
    import matplotlib.pyplot as plt
    combined_keys = list(diagrams.keys())
    n_plots = len(combined_keys)
    n_rows = int(np.ceil(n_plots / n_cols))
    base_height = base_size[1]
    base_width = base_size[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(base_width * n_cols, base_height * n_rows), dpi=dpi)
    axes = np.atleast_2d(axes).flatten()  # Ensure axs is a 1D array for easy iteration
    for i, key in enumerate(combined_keys):
        # avoid plotting empty subplots
        if i >= n_plots:
            break
        plot_persistence_diagram(diagrams[key], ax=axes[i], marker_size=marker_size, title=key, show=False, legend=legend, legend_loc=legend_loc, fontsize=fontsize, rasterized=rasterized, **kwargs)

    # Save the plot
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_betti_curve(diagram, nbins=100, homology_dimensions=[0,1,2], title="Betti curves", ymax=10, ax=None, show=True, legend=True, legend_loc='upper right', label_axes=True, axis_ticks=True, fontsize=12):
    """
    Plots Betti curves, which track the number of topological features over filtration values.

    Args:
        diagram (array-like): 
            Persistence diagram data used to compute Betti curves.
        nbins (int, optional): 
            Number of bins for filtration values. Defaults to `100`.
        homology_dimensions (list, optional): 
            List of homology dimensions to plot. Defaults to `[0, 1, 2]`.
        title (str, optional): 
            Title of the plot. Defaults to `"Betti curves"`.
        ymax (int, optional): 
            Maximum y-axis value. Defaults to `10`.
        ax (matplotlib.axes.Axes, optional): 
            Axis object for plotting. If `None`, a new figure is created.
        show (bool, optional): 
            Whether to display the plot. Defaults to `True`.
        legend (bool, optional): 
            Whether to include a legend. Defaults to `True`.
        legend_loc (str, optional): 
            Location of the legend. Defaults to `'upper right'`.
        label_axes (bool, optional): 
            Whether to label the axes. Defaults to `True`.
        axis_ticks (bool, optional): 
            Whether to include axis ticks. Defaults to `True`.
        fontsize (int, optional): 
            Font size for labels and title. Defaults to `12`.

    Returns:
        matplotlib.axes.Axes: The axis object containing the Betti curve.

    Example:
        ```python
        plot_betti_curve(diagram, nbins=50, homology_dimensions=[0,1])
        ```
    """

    from gtda.diagrams import BettiCurve
    betti_curve = BettiCurve(n_bins=nbins)
    betti_curves = betti_curve.fit_transform(diagram)
    filtration_values = betti_curve.samplings_

    # Plot Betti curves for Betti-0, Betti-1, and Betti-2
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    for dim in homology_dimensions:
        ax.plot(filtration_values[dim], betti_curves[0][dim, :], label=f'Betti-{dim}')

    if label_axes:
        ax.set_xlabel('Filtration Parameter', fontsize=fontsize-2)
        ax.set_ylabel('Betti Numbers', fontsize=fontsize-2)

    if not axis_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title, fontsize=fontsize)
    ax.set_ylim(0, ymax)
    if legend:
        ax.legend(loc=legend_loc)

    if show:
        plt.show()
    
    return ax


def plot_betti_curves(diagrams, nbins=100, ymax=8, n_cols=3, base_size=(3,3), dpi=300, legend=True, save_path=None, **kwargs):
    """
    Plots Betti curves for multiple persistence diagrams in a grid layout.

    Parameters
    ----------
    diagrams : dict
        A dictionary where keys are diagram names and values are persistence diagrams.
    nbins : int, optional
        Number of bins to use for Betti curve computation, by default 100.
    ymax : int, optional
        Maximum y-axis limit for the Betti curves, by default 8.
    n_cols : int, optional
        Number of columns in the grid layout, by default 3.
    base_size : tuple, optional
        Base figure size (width, height) for each subplot, by default (3,3).
    dpi : int, optional
        Dots per inch for figure resolution, by default 300.
    legend : bool, optional
        Whether to include a legend in each plot, by default True.
    save_path : str, optional
        File path to save the plot. If None, the plot is displayed instead.
    **kwargs : dict
        Additional keyword arguments passed to `plot_betti_curve`.

    Returns
    -------
    None
        Displays or saves the plotted figure.

    Notes
    -----
    Each subplot corresponds to a Betti curve computed from a persistence diagram.
    """
    # Plot the betti curves into a single figure
    import matplotlib.pyplot as plt
    combined_keys = list(diagrams.keys())
    n_plots = len(combined_keys)
    n_rows = int(np.ceil(n_plots / n_cols))
    base_height = base_size[1]
    base_width = base_size[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(base_width * n_cols, base_height * n_rows), dpi=dpi)
    axes = np.atleast_2d(axes).flatten()  # Ensure axs is a 1D array for easy iteration
    for i, key in enumerate(combined_keys):
        # avoid plotting empty subplots
        if i >= n_plots:
            break
        plot_betti_curve(diagrams[key], nbins=nbins, ymax=ymax, ax=axes[i], title=key, show=False, legend=legend, **kwargs)

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
    Plots a grouped bar chart of Betti number statistics across different methods.

    Args:
        betti_stats_pivot (pd.DataFrame): 
            DataFrame containing Betti number statistics.
        statistic (str, optional): 
            Statistic to plot (e.g., 'Entropy', 'Variance'). Defaults to `'Entropy'`.
        dimension (str or int, optional): 
            Specific homology dimension to plot. Defaults to `None` (plots all).
        log_y (bool, optional): 
            Whether to use a logarithmic scale on the y-axis. Defaults to `False`.
        bar_width (float, optional): 
            Width of bars in the grouped bar chart. Defaults to `0.2`.
        pal (str, optional): 
            Color palette. Defaults to `'tab20'`.
        figsize (tuple, optional): 
            Figure size in inches. Defaults to `(7, 4)`.
        dpi (int, optional): 
            Resolution in dots per inch. Defaults to `300`.
        save_path (str, optional): 
            Path to save the figure. Defaults to `None`.

    Returns:
        None

    Example:
        ```python
        plot_betti_statistic(betti_stats_df, statistic='Entropy', save_path="betti_statistic.png")
        ```
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
    Plots distance metrics for Betti numbers across different methods.

    Args:
        distance_metrics_df (pd.DataFrame): 
            DataFrame containing distance metrics.
        metric (str): 
            Metric to plot (`'L1 Distance'`, `'L2 Distance'`, `'Total Relative Error'`).
        color (str, optional): 
            Color of the bars in the plot. Defaults to `'teal'`.
        log_y (bool, optional): 
            Whether to use a logarithmic scale on the y-axis. Defaults to `False`.
        figsize (tuple, optional): 
            Figure size in inches. Defaults to `(6, 4)`.
        dpi (int, optional): 
            Resolution in dots per inch. Defaults to `300`.
        save_path (str, optional): 
            File path to save the plot. Defaults to `None`.

    Returns:
        None

    Example:
        ```python
        plot_betti_distance(distance_metrics_df, metric="L1 Distance")
        ```
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


