

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .. import logger
import math
import time

from ..utils.importance_analysis import compute_feature_importance

def visualize_importance_weights(model, adata, top_n=20, mode='histogram', fontsize=12, figsize=(5, 3), save_path=None):
    if not model.use_importance_mask:
        logger.warning("Importance mask is not used in this model.")
        return

    importance_weights = model.get_importance_weights().detach().cpu().numpy()

    if mode == 'histogram':
        plt.figure(figsize=figsize)
        plt.hist(importance_weights, bins=30, edgecolor='k')
        plt.xlabel('Importance Weight', fontsize=fontsize)
        plt.ylabel('Frequency', fontsize=fontsize)
        plt.title('Distribution of Importance Weights', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    else:
        if mode == 'highest':
            top_indices = np.argsort(importance_weights)[-top_n:]
        elif mode == 'lowest':
            top_indices = np.argsort(importance_weights)[:top_n]
        elif mode == 'absolute':
            top_indices = np.argsort(np.abs(importance_weights))[-top_n:]
        else:
            raise ValueError("Mode must be one of ['highest', 'lowest', 'absolute', 'histogram']")

        top_weights = importance_weights[top_indices]
        feature_names = adata.var_names[top_indices]

        plt.figure(figsize=figsize)
        plt.barh(range(top_n), top_weights, align='center')
        plt.yticks(range(top_n), feature_names, fontsize=fontsize)
        plt.xlabel('Importance Weight', fontsize=fontsize)
        plt.ylabel('Feature Names', fontsize=fontsize)
        plt.title(f'Top Features by Importance Weight ({mode})', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()




def plot_importance_heatmap(importance_matrix, input_feature=None, figsize=(20, 15), save_path=None):
    """
    Plots a heatmap of the importance matrix with adjustments for label readability.

    Parameters:
    - importance_matrix (torch.Tensor): The importance matrix with shape (n_input_features, n_encoded_neurons).
    - adata (anndata.AnnData): The AnnData object containing the input features.
    """

    # Extract input feature names from adata
    input_feature_names = input_feature
    encoded_neuron_names = [f'Neuron {i}' for i in range(importance_matrix.shape[1])]

    # Create a DataFrame for the heatmap
    df_importance = pd.DataFrame(importance_matrix.T, index=encoded_neuron_names, columns=input_feature_names)

    # Plot the heatmap with hierarchical clustering
    cluster_grid = sns.clustermap(
        df_importance,
        cmap='viridis',
        annot=False,
        cbar=True,
        figsize=figsize,
        #xticklabels=True,
        yticklabels=True
    )

    # Adjust the x-axis labels for better readability
    plt.setp(cluster_grid.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
    plt.setp(cluster_grid.ax_heatmap.yaxis.get_majorticklabels(), fontsize=10)

    # Adjust the overall plot to make room for labels
    cluster_grid.figure.subplots_adjust(bottom=0.3, right=0.8)

    plt.title('Feature Importance Heatmap with Hierarchical Clustering')
    plt.xlabel('Input Features')
    plt.ylabel('Encoded Neurons')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_top_genes_per_neuron(ranked_gene_lists, show_neurons=None, top_n=10, ncols=4, figsize=(4, 4), save_path=None):
    """
    Plots bar charts of the top contributing genes for each neuron in a compact grid layout using pre-ranked lists.

    Parameters:
    - ranked_gene_lists (dict): A dictionary where keys are neuron names and values are DataFrames containing ranked genes.
    - show_neurons (list): List of neurons to plot. If None, plots all neurons in the dictionary.
    - top_n (int): Number of top contributing genes to display for each neuron.
    - ncols (int): Number of columns in the grid layout.
    - figsize (tuple): Size of each subplot (width, height).
    - save_path (str): File path to save the plot. If None, displays the plot.
    """

    # If `show_neurons` is None, use all available neurons
    if show_neurons is None:
        show_neurons = list(ranked_gene_lists.keys())
    else:
        # Filter the provided neurons to only those in `ranked_gene_lists`
        show_neurons = [neuron for neuron in show_neurons if neuron in ranked_gene_lists]

    # Determine the number of rows needed
    nrows = math.ceil(len(show_neurons) / ncols)

    # Create subplots
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize[0], nrows * figsize[1]),
        constrained_layout=True,
    )
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Plot top genes for each neuron
    for idx, neuron in enumerate(show_neurons):
        if neuron in ranked_gene_lists:
            top_genes = ranked_gene_lists[neuron].head(top_n)
            sns.barplot(
                x=top_genes["Importance"].values,
                y=top_genes["Gene"].values,
                palette="viridis_r",
                ax=axes[idx],
            )
            axes[idx].set_title(f"Top {top_n} Contributing Genes for {neuron}")
            axes[idx].set_xlabel("Importance")
            axes[idx].set_ylabel("Genes")

    # Remove empty subplots
    for i in range(len(show_neurons), len(axes)):
        fig.delaxes(axes[i])

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

