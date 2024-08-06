
import gseapy as gp
import pandas as pd
import os
from ..plotting.pl_enrichment import plot_go_enrichment


def compute_go(feature_list, organism="human", top_n=10, qval_correct=1e-10, color_palette='viridis_r', font_size=12, dpi=300, figsize=(10,3), save_path=None):
    gp_results = gp.enrichr(gene_list=feature_list, gene_sets='GO_Biological_Process_2021', organism=organism, outdir=None)
    plot_go_enrichment(gp_results, top_n=top_n, qval_correct=qval_correct, color_palette=color_palette, font_size=font_size, figsize=figsize, dpi=dpi, save_path=save_path)
    return gp_results

def run_gsea_for_all_neurons(ranked_lists, gene_sets='GO_Biological_Process_2021', outdir='GSEA_results',
                             processes = 4, permutation_num=500, seed=0):
    """
    Run GSEA enrichment analysis for all neurons and save the results.

    Parameters:
    - ranked_lists (dict): A dictionary with neuron names as keys and ranked gene lists as values.
    - gene_sets (str): Name of the gene set database to use.
    - outdir (str): Directory to save the results.

    Returns:
    - all_gsea_results (dict): A dictionary with neuron names as keys and their GSEA results as values.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    all_gsea_results = {}

    for neuron_name, ranked_list in ranked_lists.items():
        print(f"Running GSEA for {neuron_name}...")
        neuron_outdir = os.path.join(outdir, neuron_name.replace(" ", "_"))
        if not os.path.exists(neuron_outdir):
            os.makedirs(neuron_outdir)

        # Run GSEA
        gsea_results = gp.prerank(
            rnk=ranked_list,
            gene_sets=gene_sets,
            processes=processes,
            permutation_num=permutation_num,
            outdir=neuron_outdir,
            format='png',
            seed=seed,
            min_size=10,
            max_size=1000,
        )

        all_gsea_results[neuron_name] = gsea_results

    return all_gsea_results


def get_gsea_tables(all_gsea_results):
    """
    Extract the res2d DataFrame from the GSEA results for each neuron.

    Parameters:
    - all_gsea_results (dict): A dictionary with neuron names as keys and their GSEA results as values.

    Returns:
    - extracted_results (dict): A dictionary with neuron names as keys and their res2d DataFrame as values.
    """
    res_tbls = {}

    for neuron_name, gsea_result in all_gsea_results.items():
        res_tbls[neuron_name] = gsea_result.res2d

    return res_tbls