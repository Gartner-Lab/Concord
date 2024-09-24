import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import math
import os


def plot_go_enrichment(gp_results, top_n=10, qval_correct=1e-10, color_palette='viridis_r', font_size=12, figsize=(7,3), dpi=300, save_path=None):
    """Plot top GO enrichment terms."""
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
            file_suffix = f"{time.strftime('%b%d-%H%M')}"
            plt.savefig(f"{save_path}iff_top_enriched_terms_{file_suffix}.png")
        else:
            plt.show()
        plt.close()


def plot_all_top_enriched_terms(all_gsea_results, top_n=10, ncols=1, font_size=10,
                            color_palette='viridis_r', qval_correct=1e-10,
                                figsize=(4, 4), dpi=300, save_path=None):
    """
    Plot the top enriched terms for each neuron.

    Parameters:
    - all_gsea_results (dict): A dictionary with neuron names as keys and their GSEA results as values.
    - top_n (int): Number of top enriched terms to display for each neuron.
    - ncols (int): Number of columns in the grid layout.
    - figsize (tuple): The size of each subplot.
    - font_size (int): The font size for titles and labels.
    - color_palette (str): The color palette to use for significance coloring.

    Returns:
    None
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
    Plot GSEA results for all neurons in a compact view.

    Parameters:
    - all_gsea_results (dict): A dictionary with neuron names as keys and their GSEA results as values.
    - terms_per_plot (int): Number of top enriched terms to display for each neuron.
    - ncols (int): Number of columns in the grid layout.
    - figsize_per_plot (tuple): Size of each subplot.

    Returns:
    None
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