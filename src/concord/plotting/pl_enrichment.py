import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import math
import os


def plot_go_enrichment(gp_results, top_n=10, qval_correct=1e-10, color_palette='viridis_r', font_size=12, figsize=(7,3), dpi=300, save_path=None):
    """
    Plots the top Gene Ontology (GO) enrichment terms based on adjusted p-values (FDR q-values).

    Args:
        gp_results (object): 
            GO enrichment results object containing a DataFrame in `gp_results.results`.
        top_n (int, optional): 
            Number of top terms to display. Defaults to `10`.
        qval_correct (float, optional): 
            A small correction factor added to q-values before taking `-log10`. Defaults to `1e-10`.
        color_palette (str, optional): 
            Color palette for the bar plot. Defaults to `'viridis_r'`.
        font_size (int, optional): 
            Font size for plot labels. Defaults to `12`.
        figsize (tuple, optional): 
            Size of the figure in inches (width, height). Defaults to `(7,3)`.
        dpi (int, optional): 
            Dots per inch (resolution) for saving the figure. Defaults to `300`.
        save_path (str, optional): 
            File path to save the figure. If `None`, the figure is displayed instead of being saved. Defaults to `None`.

    Returns:
        None

    Example:
        ```python
        plot_go_enrichment(gp_results, top_n=15, save_path="go_enrichment.png")
        ```
    """
    if gp_results is not None:
        top_terms = gp_results.results[['Term', 'Adjusted P-value']].rename(columns={'Adjusted P-value': 'FDR q-val'})
        top_terms = top_terms.nsmallest(top_n, 'FDR q-val')
        top_terms['-log10(FDR q-val)'] = -np.log10(top_terms['FDR q-val'] + qval_correct)

        top_terms = top_terms.sort_values(by='-log10(FDR q-val)', ascending=False)
        print(figsize)
        plt.figure(figsize=figsize, dpi=dpi)
        sns.barplot(x='-log10(FDR q-val)', y='Term', data=top_terms, palette=color_palette)
        plt.title(f'Top {top_n} Enriched Terms', fontsize=font_size + 2)
        plt.xlabel('-log10(FDR q-val)', fontsize=font_size)
        plt.ylabel('Enriched Terms', fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.show()
        plt.close()


def plot_all_top_enriched_terms(all_gsea_results, top_n=10, ncols=1, font_size=10,
                            color_palette='viridis_r', qval_correct=1e-10,
                                figsize=(4, 4), dpi=300, save_path=None):
    """
    Plots the top enriched Gene Set Enrichment Analysis (GSEA) terms for multiple neurons.

    Args:
        all_gsea_results (dict): 
            Dictionary where keys are neuron names and values are GSEA results DataFrames.
        top_n (int, optional): 
            Number of top enriched terms to display per neuron. Defaults to `10`.
        ncols (int, optional): 
            Number of columns in the subplot grid layout. Defaults to `1`.
        font_size (int, optional): 
            Font size for plot labels. Defaults to `10`.
        color_palette (str, optional): 
            Color palette for the bar plots. Defaults to `'viridis_r'`.
        qval_correct (float, optional): 
            A small correction factor added to q-values before taking `-log10`. Defaults to `1e-10`.
        figsize (tuple, optional): 
            Size of each subplot (width, height) in inches. Defaults to `(4,4)`.
        dpi (int, optional): 
            Resolution of the output figure. Defaults to `300`.
        save_path (str, optional): 
            File path to save the figure. If `None`, the figure is displayed. Defaults to `None`.

    Returns:
        None

    Example:
        ```python
        plot_all_top_enriched_terms(all_gsea_results, top_n=5, ncols=2, save_path="gsea_terms.pdf")
        ```
    """
    n_neurons = len(all_gsea_results)
    nrows = math.ceil(n_neurons / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * figsize[0], nrows * figsize[1]),
                             constrained_layout=True, dpi=dpi)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    for idx, (neuron_name, gsea_results) in enumerate(all_gsea_results.items()):
        # Ensure NES and FDR q-val are numeric
        gsea_results['NES'] = pd.to_numeric(gsea_results['NES'], errors='coerce')
        gsea_results['FDR q-val'] = pd.to_numeric(gsea_results['FDR q-val'], errors='coerce')

        # Select top enriched terms with positive NES
        positive_terms = gsea_results[gsea_results['NES'] > 0]
        top_terms = positive_terms.nsmallest(top_n, 'FDR q-val')
        top_terms['-log10(FDR q-val)'] = -np.log10(top_terms['FDR q-val']+qval_correct)

        top_terms = top_terms.sort_values(by='-log10(FDR q-val)', ascending=False)

        # Plot barplot
        sns.barplot(x='-log10(FDR q-val)', y='Term', data=top_terms, palette=color_palette, ax=axes[idx])

        axes[idx].set_title(f'Top {top_n} Enriched Terms for {neuron_name}', fontsize=font_size + 2)
        axes[idx].set_xlabel('-log10(FDR q-val)', fontsize=font_size)
        axes[idx].set_ylabel('Enriched Terms', fontsize=font_size)
        axes[idx].tick_params(axis='both', which='major', labelsize=font_size)

    # Remove empty subplots
    for i in range(len(all_gsea_results), len(axes)):
        fig.delaxes(axes[i])

    #plt.subplots_adjust(top=0.6, bottom=0.4)

    if save_path:
        file_suffix = f"{time.strftime('%b%d-%H%M')}"
        save_path = f"{save_path}_{file_suffix}.pdf"
        plt.savefig(save_path)
    else:
        plt.show()





def plot_all_top_gsea_results(all_gsea_results, terms_per_plot=5, ncols=4, figsize_per_plot=(3, 4), dpi=300, save_path=None):
    """
    Plots Gene Set Enrichment Analysis (GSEA) results for multiple neurons in a grid layout.

    Args:
        all_gsea_results (dict): 
            Dictionary where keys are neuron names and values are GSEA result objects.
        terms_per_plot (int, optional): 
            Number of top enriched terms to display per neuron. Defaults to `5`.
        ncols (int, optional): 
            Number of columns in the subplot grid. Defaults to `4`.
        figsize_per_plot (tuple, optional): 
            Size of each subplot (width, height) in inches. Defaults to `(3,4)`.
        dpi (int, optional): 
            Resolution of the output figure in dots per inch. Defaults to `300`.
        save_path (str, optional): 
            File path to save the figure. If `None`, the figure is displayed. Defaults to `None`.

    Returns:
        None

    Example:
        ```python
        plot_all_top_gsea_results(all_gsea_results, terms_per_plot=7, ncols=3, save_path="gsea_results.png")
        ```
    """

    from PIL import Image
    n_neurons = len(all_gsea_results)
    nrows = math.ceil(n_neurons / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols * figsize_per_plot[0], nrows * figsize_per_plot[1]),
                             dpi=dpi,
                             constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    tmp_dir = "gsea_tmp_plots"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    for idx, (neuron_name, gsea_results) in enumerate(all_gsea_results.items()):
        terms = gsea_results.res2d.Term[:terms_per_plot]
        # Plot the GSEA results and save the figure
        gsea_fig = gsea_results.plot(terms=terms,
                                     show_ranking=True,
                                     figsize=figsize_per_plot)

        plot_path = os.path.join(tmp_dir, f"{neuron_name}.png")
        gsea_fig.savefig(plot_path)
        plt.close(gsea_fig)

        # Load the saved figure and draw it into the subplot
        img = Image.open(plot_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(neuron_name)

    # Remove empty subplots
    for i in range(len(all_gsea_results), len(axes)):
        fig.delaxes(axes[i])

    if save_path:
        file_suffix = f"{time.strftime('%b%d-%H%M')}"
        save_path = f"{save_path}_{file_suffix}.png"
        plt.savefig(save_path)
    else:
        plt.show()

    # Clean up temporary files
    for plot_file in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, plot_file))
    os.rmdir(tmp_dir)