import matplotlib.pyplot as plt
import scanpy as sc
import warnings
import numpy as np
import wandb
import io
import plotly.express as px
import plotly.io as pio
import logging

logger = logging.getLogger(__name__)

class Plotter:
    def __init__(self):
        pass

    def plot_embeddings(self, adata, show_emb, show_cols, figsize=(9, 3), dpi=300, ncols=1,
                        font_size=8, point_size=10, legend_loc='on data', save_path=None):
        """
        Plot embeddings from an AnnData object with customizable parameters.

        Parameters:
        adata (anndata.AnnData): The AnnData object.
        show_emb (str): The embedding to plot.
        show_cols (list of str): The columns to color the embeddings by.
        figsize (tuple): The size of the figure.
        dpi (int): The resolution of the figure.
        ncols (int): The number of columns in the subplot grid.
        font_size (int): The font size for titles and labels.
        point_size (int): The size of the points in the plot.

        Returns:
        None
        """
        # Calculate the number of rows needed
        nrows = int(np.ceil(len(show_cols) / ncols))

        # Create subplots
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
        axs = np.atleast_2d(axs).flatten()  # Ensure axs is a 1D array for easy iteration

        warnings.filterwarnings('ignore')
        for col, ax in zip(show_cols, axs):
            sc.pl.embedding(adata, basis=show_emb, color=col, ax=ax, show=False,
                            legend_loc=legend_loc, legend_fontsize=font_size, size=point_size)
            ax.set_title(ax.get_title(), fontsize=font_size)  # Adjust plot title font size
            ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)  # Adjust X-axis label font size
            ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)  # Adjust Y-axis label font size
            cbar = ax.collections[-1].colorbar
            if cbar is not None:
                cbar.ax.tick_params(labelsize=font_size)

        # Turn off any unused axes
        for ax in axs[len(show_cols):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi)

    def visualize_3d_embedding(self, adata, embedding_key='encoded_UMAP', color_by='batch', save_path=None, point_size=3,
                               opacity=0.7, width=800, height=600):
        """
        Visualize 3D embedding and color by selected columns in adata.obs, with options for plot aesthetics and saving the plot.

        Parameters:
        adata : AnnData
            AnnData object containing embeddings and metadata.
        embedding_key : str, optional
            Key in adata.obsm where the embedding is stored. Default is 'encoded_UMAP'.
        color_by : str, optional
            Column in adata.obs to color the points by. Default is 'batch'.
        save_path : str, optional
            Path to save the plot. If None, the plot will not be saved.
        point_size : int, optional
            Size of the points in the plot. Default is 3.
        opacity : float, optional
            Opacity of the points in the plot. Default is 0.7.
        width : int, optional
            Width of the plot in pixels. Default is 800.
        height : int, optional
            Height of the plot in pixels. Default is 600.

        Returns:
        fig : plotly.graph_objects.Figure
            Plotly figure object.
        """
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

        # Plotly 3D scatter plot
        fig = px.scatter_3d(df, x='DIM1', y='DIM2', z='DIM3', color=color_by,
                            title=f'3D Embedding colored by {color_by}', labels={'color': color_by},
                            opacity=opacity)

        fig.update_traces(marker=dict(size=point_size), selector=dict(mode='markers'))
        fig.update_layout(width=width, height=height)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"3D plot saved to {save_path}")

        fig.show()
        return fig

    def visualize_importance_weights(self, model, adata, top_n=20, mode='histogram', save_path=None):
        if not model.use_importance_mask:
            logger.warning("Importance mask is not used in this model.")
            return

        importance_weights = model.get_importance_weights().detach().cpu().numpy()

        if mode == 'histogram':
            plt.figure(figsize=(10, 6))
            plt.hist(importance_weights, bins=30, edgecolor='k')
            plt.xlabel('Importance Weight')
            plt.ylabel('Frequency')
            plt.title('Distribution of Importance Weights')

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

            plt.figure(figsize=(10, 6))
            plt.barh(range(top_n), top_weights, align='center')
            plt.yticks(range(top_n), feature_names)
            plt.xlabel('Importance Weight')
            plt.ylabel('Feature Names')
            plt.title(f'Top Features by Importance Weight ({mode})')

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

    def wandb_log_plot(self, plt, plot_name, plot_path=None):
        """
        Logs a confusion matrix plot to W&B. If plot_path is provided, saves the plot to the specified path before logging.
        Otherwise, logs the plot directly from an in-memory buffer.

        Args:
            plt (matplotlib.pyplot): The Matplotlib pyplot object with the confusion matrix plot.
            plot_path (str, optional): The file path to save the plot. If None, the plot is logged from an in-memory buffer.
        """
        if plot_path is not None:
            plt.savefig(str(plot_path))
            wandb.log({plot_name: wandb.Image(str(plot_path))})
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            wandb.log({plot_name: wandb.Image(buf)})


