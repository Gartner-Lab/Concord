

import os
from ..plotting.pl_enrichment import plot_go_enrichment


def compute_go(feature_list, organism="human", top_n=10, qval_correct=1e-10, color_palette='viridis_r', font_size=12, dpi=300, figsize=(10,3), save_path=None):
    """
    Performs Gene Ontology (GO) enrichment analysis using gseapy.enrichr and visualizes the results.

    Args:
        feature_list (list): List of gene symbols to analyze.
        organism (str, optional): Organism name (e.g., "human", "mouse"). Defaults to "human".
        top_n (int, optional): Number of top enriched GO terms to plot. Defaults to 10.
        qval_correct (float, optional): Maximum adjusted p-value for filtering significant terms. Defaults to 1e-10.
        color_palette (str, optional): Color palette for the enrichment plot. Defaults to 'viridis_r'.
        font_size (int, optional): Font size for plot text. Defaults to 12.
        dpi (int, optional): Resolution of the output plot. Defaults to 300.
        figsize (tuple, optional): Figure size for the plot (width, height). Defaults to (10,3).
        save_path (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.

    Returns:
        dict: Enrichment results from `gseapy.enrichr`.
    """

    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy is required for this method. Please install it using 'pip install gseapy'.")

    gp_results = gp.enrichr(gene_list=feature_list, gene_sets='GO_Biological_Process_2021', organism=organism, outdir=None)
    plot_go_enrichment(gp_results, top_n=top_n, qval_correct=qval_correct, color_palette=color_palette, font_size=font_size, figsize=figsize, dpi=dpi, save_path=save_path)
    return gp_results

def run_gsea_for_all_neurons(ranked_lists, gene_sets='GO_Biological_Process_2021', outdir='GSEA_results',
                             processes = 4, permutation_num=500, seed=0):
    """
    Runs Gene Set Enrichment Analysis (GSEA) for multiple neuron types and saves results.

    Args:
        ranked_lists (dict): Dictionary where keys are neuron names and values are ranked gene lists.
        gene_sets (str, optional): Name of the gene set database to use (e.g., 'GO_Biological_Process_2021'). Defaults to 'GO_Biological_Process_2021'.
        outdir (str, optional): Directory to save GSEA results. Defaults to 'GSEA_results'.
        processes (int, optional): Number of parallel processes for GSEA computation. Defaults to 4.
        permutation_num (int, optional): Number of gene set permutations for statistical significance. Defaults to 500.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        dict: A dictionary where keys are neuron names and values are their respective GSEA results.
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("gseapy is required for this method. Please install it using 'pip install gseapy'.")

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
    Extracts the GSEA summary tables from the enrichment results.

    Args:
        all_gsea_results (dict): Dictionary where keys are neuron names and values are GSEA results.

    Returns:
        dict: A dictionary where keys are neuron names and values are their respective `res2d` DataFrame containing enrichment statistics.
    """
    res_tbls = {}

    for neuron_name, gsea_result in all_gsea_results.items():
        res_tbls[neuron_name] = gsea_result.res2d

    return res_tbls