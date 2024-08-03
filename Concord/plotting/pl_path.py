

import numpy as np
import matplotlib.pyplot as plt

def plot_signal_over_path(sorted_data, sorted_smoothed_data, x, ncols=4, same_yaxis_range=False):
    """
    Plot geodesic distances for each dimension with optional same y-axis range for all plots.

    Parameters:
    sorted_data (ndarray): Original sorted data array.
    sorted_smoothed_data (ndarray): Smoothed data array.
    geodesic_distances (ndarray): Geodesic distances.
    ncols (int): Number of columns in the subplot grid.
    same_yaxis_range (bool): Whether to use the same y-axis range for all plots.

    Returns:
    None
    """
    n_dims = sorted_smoothed_data.shape[1]
    nrows = int(np.ceil(n_dims / ncols))

    # Create the subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)

    # Ensure axs is a 1D array for easy iteration
    axs = axs.flatten()

    # Determine y-axis range if same_yaxis_range is True
    if same_yaxis_range:
        y_min = np.min([sorted_data.min(), sorted_smoothed_data.min()])
        y_max = np.max([sorted_data.max(), sorted_smoothed_data.max()])

    for i in range(n_dims):
        ax = axs[i]
        ax.plot(x, sorted_data[:, i], color='blue', label='Original Sorted')
        ax.plot(x, sorted_smoothed_data[:, i], color='red', linestyle='--', label='Smoothed')
        ax.set_title(f'Dimension {i}')
        ax.set_xlabel('Geodesic distance to Point A')
        ax.set_ylabel('Value')
        ax.legend()

        # Set the same y-axis range for all plots if required
        if same_yaxis_range:
            ax.set_ylim(y_min, y_max)

    # Turn off any unused axes
    for j in range(n_dims, len(axs)):
        axs[j].axis('off')

    plt.show()
