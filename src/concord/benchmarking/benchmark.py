from __future__ import annotations
import pandas as pd
import logging
import numpy as np
import logging
from pathlib import Path
from typing import Sequence, Mapping, Literal, Optional
import numpy as np
import pandas as pd
import pickle

from ..plotting import plot_benchmark_table
from .tda import compute_persistent_homology


logger = logging.getLogger(__name__)

def count_total_runs(param_grid):
    total_runs = 0
    for key, values in param_grid.items():
        if isinstance(values, list):
            total_runs += len(values)
    return total_runs


def compute_correlation(data_dict, corr_types=['pearsonr', 'spearmanr', 'kendalltau'], groundtruth_key="PCA_no_noise"):
    from scipy.stats import pearsonr, spearmanr, kendalltau
    import pandas as pd

    pr_result, sr_result, kt_result = {}, {}, {}
    
    # Calculate correlations based on requested types
    for key in data_dict.keys():
        ground_val = data_dict[groundtruth_key]
        ground_val = np.array(list(ground_val.values())) if isinstance(ground_val, dict) else ground_val

        latent_val = data_dict[key]
        latent_val = np.array(list(latent_val.values())) if isinstance(latent_val, dict) else latent_val

        if 'pearsonr' in corr_types:
            pr_result[key] = pearsonr(ground_val, latent_val)[0]
        if 'spearmanr' in corr_types:
            sr_result[key] = spearmanr(ground_val, latent_val)[0]
        if 'kendalltau' in corr_types:
            kt_result[key] = kendalltau(ground_val, latent_val)[0]
    
    # Collect correlation values for each type
    corr_values = {}
    for key in data_dict.keys():
        corr_values[key] = [
            pr_result.get(key, None),
            sr_result.get(key, None),
            kt_result.get(key, None)
        ]
    
    # Create DataFrame with correlation types as row indices and keys as columns
    corr_df = pd.DataFrame(corr_values, index=['pearsonr', 'spearmanr', 'kendalltau'])

    return corr_df.T


def compare_graph_connectivity(adata, emb1, emb2, k=30, use_faiss=False, use_ivf=False, ivf_nprobe=10, metric=['jaccard', 'frobenius', 'hamming'], dist_metric='euclidean'):
    """
    Compare the graph connectivity of two embeddings by computing their k-NN graphs
    and comparing their adjacency matrices using specified metrics.

    Parameters:
    - adata: AnnData
        AnnData object containing embeddings in `adata.obsm`.
    - emb1: str
        Key for the first embedding in `adata.obsm`.
    - emb2: str
        Key for the second embedding in `adata.obsm`.
    - k: int
        Number of nearest neighbors for the k-NN graph.
    - use_faiss: bool
        Whether to use FAISS for nearest neighbor computation.
    - use_ivf: bool
        Whether to use IVF FAISS index.
    - ivf_nprobe: int
        Number of probes for IVF FAISS index.
    - metric: list of str
        List of metrics to use for graph comparison: ['jaccard', 'frobenius', 'hamming'].

    Returns:
    - graph_distance: dict
        Dictionary with keys as metric names and values as similarity scores.
    """
    from scipy.sparse import csr_matrix
    import numpy as np
    from ..model.knn import Neighborhood  # Adjust import based on your directory structure

    # Check if embeddings exist in adata.obsm
    if emb1 not in adata.obsm or emb2 not in adata.obsm:
        raise ValueError(f"Embedding keys {emb1} and {emb2} not found in adata.obsm.")
    
    emb1 = adata.obsm[emb1]
    emb2 = adata.obsm[emb2]

    # Initialize Neighborhood objects for both embeddings
    neighborhood1 = Neighborhood(emb1, k=k, use_faiss=use_faiss, use_ivf=use_ivf, ivf_nprobe=ivf_nprobe, metric=dist_metric)
    neighborhood2 = Neighborhood(emb2, k=k, use_faiss=use_faiss, use_ivf=use_ivf, ivf_nprobe=ivf_nprobe, metric=dist_metric)

    # Compute k-NN indices for all points
    core_samples1 = np.arange(emb1.shape[0])
    core_samples2 = np.arange(emb2.shape[0])

    indices1 = neighborhood1.get_knn(core_samples1, k=k, include_self=False)
    indices2 = neighborhood2.get_knn(core_samples2, k=k, include_self=False)

    # Create adjacency matrices
    rows1 = np.repeat(core_samples1, k)
    cols1 = indices1.flatten()
    graph1 = csr_matrix((np.ones_like(cols1), (rows1, cols1)), shape=(emb1.shape[0], emb1.shape[0]))

    rows2 = np.repeat(core_samples2, k)
    cols2 = indices2.flatten()
    graph2 = csr_matrix((np.ones_like(cols2), (rows2, cols2)), shape=(emb2.shape[0], emb2.shape[0]))

    # Compare graphs based on the chosen metric
    graph_distance = {}
    if 'jaccard' in metric:
        graph1_binary = graph1 > 0
        graph2_binary = graph2 > 0
        intersection = graph1_binary.multiply(graph2_binary).sum()
        union = (graph1_binary + graph2_binary > 0).sum()
        graph_distance['jaccard'] = intersection / union
    if 'hamming' in metric:
        graph1_binary = graph1 > 0
        graph2_binary = graph2 > 0
        graph_distance['hamming'] = 1 - (graph1_binary != graph2_binary).sum() / graph1_binary.nnz
    if 'frobenius' in metric:
        graph_distance['frobenius'] = np.linalg.norm((graph1 - graph2).toarray())

    return graph_distance


def benchmark_graph_connectivity(adata, emb_keys, k=30, use_faiss=False, use_ivf=False, ivf_nprobe=10, metric=['jaccard', 'hamming'], 
                                 groundtruth_keys = {'(nn)': 'PCA_no_noise','(wn)': 'PCA_wt_noise'}, dist_metric='cosine'):

    connectivity_df = pd.DataFrame()
    for gname,gemb in groundtruth_keys.items():
        results = []
        for key in emb_keys:
            similarity_scores = compare_graph_connectivity(
                adata,
                emb1=key,
                emb2=gemb,
                k=k,
                metric=metric,
                dist_metric=dist_metric,
                use_faiss=use_faiss,
                use_ivf=use_ivf,
                ivf_nprobe=ivf_nprobe
            )
            results.append(similarity_scores)

        df = pd.DataFrame(results, index=emb_keys)
        # Add a second level index to the column named 'metric'
        df.columns = pd.MultiIndex.from_tuples([(f'Graph connectivity', col + gname) for col in df.columns])
        connectivity_df = pd.concat([connectivity_df, df], axis=1)
    
    return connectivity_df



def benchmark_topology(diagrams, expected_betti_numbers=[1,0,0], n_bins=100, save_dir=None, file_suffix=None):
    """
    Benchmark the topological properties of persistence diagrams.

    Args:
        diagrams : dict
            A dictionary where keys are method names and values are persistence diagrams.
        expected_betti_numbers : list, optional
            A list specifying the expected Betti numbers for different homology dimensions. Default is [1, 0, 0].
        n_bins : int, optional
            Number of bins to use for Betti curve calculations. Default is 100.
        save_dir : str, optional
            Directory to save benchmarking results as CSV files. If None, results are not saved.
        file_suffix : str, optional
            Suffix to append to saved filenames.

    Returns:
        dict
            A dictionary containing:
            - `'betti_stats'`: DataFrame summarizing Betti statistics.
            - `'distance_metrics'`: DataFrame of computed distances between Betti curves.
            - `'combined_metrics'`: DataFrame of entropy, variance, and L1 distance metrics.
    """
    import pandas as pd
    from .tda import compute_betti_statistics, summarize_betti_statistics

    results = {}
    betti_stats = {}    
    # Compute betti stats for all keys
    for key in diagrams.keys():
        betti_stats[key] = compute_betti_statistics(diagram=diagrams[key], expected_betti_numbers=expected_betti_numbers, n_bins=n_bins)

    betti_stats_pivot, distance_metrics_df = summarize_betti_statistics(betti_stats)
    results['betti_stats'] = betti_stats_pivot
    results['distance_metrics'] = distance_metrics_df

    entropy_columns = betti_stats_pivot.loc[:, pd.IndexSlice[:, 'Entropy']]
    average_entropy = entropy_columns.mean(axis=1)
    variance_columns = betti_stats_pivot.loc[:, pd.IndexSlice[:, 'Variance']]
    average_variance = variance_columns.mean(axis=1)
    final_metrics = pd.concat([average_entropy, average_variance], axis=1)
    final_metrics.columns = pd.MultiIndex.from_tuples([('Betti curve', 'Entropy'), ('Betti curve', 'Variance')])
    final_metrics[('Betti number', 'L1 distance')] = distance_metrics_df['L1 Distance']
    results['combined_metrics'] = final_metrics

    if save_dir is not None and file_suffix is not None:
        for key, result in results.items():
            if isinstance(result, pd.DataFrame):
                result.to_csv(save_dir / f"{key}_{file_suffix}.csv")
            else:
                continue

    return results



def benchmark_geometry(adata, keys, 
                       eval_metrics=['pseudotime', 'cell_distance_corr', 'local_distal_corr', 'trustworthiness', 'state_distance_corr', 'state_dispersion_corr', 'state_batch_distance_ratio'],
                       dist_metric='cosine', 
                       groundtruth_key = 'PCA_no_noise', 
                       state_key = 'cluster',
                       batch_key = 'batch',
                       groundtruth_dispersion = None,
                       ground_truth_dispersion_key = 'wt_noise',
                       corr_types = ['pearsonr', 'spearmanr', 'kendalltau'], 
                       trustworthiness_n_neighbors = np.arange(10, 101, 10),
                       dispersion_metric='var',
                       return_type='dataframe',
                       local_percentile=0.1,
                       distal_percentile=0.9,
                       start_point=0,
                       end_point=None,
                       pseudotime_k = 30,
                       truetime_key = 'time',
                       verbose=True,
                       save_dir=None, 
                       file_suffix=None):
    """
    Benchmark the geometric properties of different embeddings.

    Args:
        adata : anndata.AnnData
            The AnnData object containing cell embeddings.
        keys : list
            List of embeddings (keys in `adata.obsm`) to evaluate.
        eval_metrics : list, optional
            Metrics to compute, such as 'pseudotime', 'cell_distance_corr', etc. Default includes multiple metrics.
        dist_metric : str, optional
            Distance metric for computing cell distances. Default is 'cosine'.
        groundtruth_key : str, optional
            Key in `adata.obsm` containing the ground truth embedding. Default is 'PCA_no_noise'.
        state_key : str, optional
            Key in `adata.obs` representing cell states or clusters.
        batch_key : str, optional
            Key in `adata.obs` representing batch information.
        groundtruth_dispersion : dict, optional
            Precomputed dispersion values for ground truth, if available.
        ground_truth_dispersion_key : str, optional
            Key used when computing dispersion correlations. Default is 'wt_noise'.
        corr_types : list, optional
            List of correlation methods to compute. Default includes 'pearsonr', 'spearmanr', and 'kendalltau'.
        trustworthiness_n_neighbors : np.ndarray, optional
            Range of neighborhood sizes for trustworthiness computation. Default is `np.arange(10, 101, 10)`.
        dispersion_metric : str, optional
            Metric to compute dispersion, e.g., 'var' (variance). Default is 'var'.
        return_type : str, optional
            If 'dataframe', returns summary statistics; if 'full', returns additional details. Default is 'dataframe'.
        local_percentile : float, optional
            Percentile threshold for local distance correlations. Default is 0.1.
        distal_percentile : float, optional
            Percentile threshold for distal distance correlations. Default is 0.9.
        start_point : int, optional
            Index of the starting cell for pseudotime computation. Must be specified.
        end_point : int, optional
            Index of the ending cell for pseudotime computation. Must be specified.
        pseudotime_k : int, optional
            Number of neighbors used in k-NN graph for pseudotime computation. Default is 30.
        truetime_key : str, optional
            Key in `adata.obs` representing ground truth time. Default is 'time'.
        verbose : bool, optional
            Whether to enable logging. Default is True.
        save_dir : str, optional
            Directory to save benchmarking results. If None, results are not saved.
        file_suffix : str, optional
            Suffix for saved filenames.

    Returns:
        pd.DataFrame or tuple
            If `return_type='dataframe'`, returns a DataFrame summarizing benchmark results.
            If `return_type='full'`, returns both the DataFrame and a detailed results dictionary.
    """
    import pandas as pd
    from .geometry import pairwise_distance, local_vs_distal_corr, compute_trustworthiness, compute_centroid_distance, compute_state_batch_distance_ratio, compute_dispersion_across_states
    results_df = {}
    results_full = {}
    if verbose:
        logger.setLevel(logging.INFO)


    # Pseudotime correlation
    if 'pseudotime' in eval_metrics:
        from ..utils.path_analysis import shortest_path_on_knn_graph, compute_pseudotime_from_shortest_path
        logger.info("Computing pseudotime correlation")
        if start_point is None or end_point is None:
            raise ValueError("start_point and end_point must be specified for pseudotime computation.")
        if truetime_key not in adata.obs:
            raise ValueError(f"Groundtruth time key '{truetime_key}' not found in adata.obs.")
        # Compute pseudotime for each method post integration
        time_dict={}
        path_dict = {}
        for basis in keys:
            logger.info(f"Computing pseudotime for {basis}")
            if basis not in adata.obsm:
                continue
            
            pseudotime_key = f"{basis}_pseudotime"
            try:
                path, _ = shortest_path_on_knn_graph(adata, emb_key=basis, k=pseudotime_k, point_a=start_point, point_b=end_point, use_faiss=True)
                time_dict[basis] = compute_pseudotime_from_shortest_path(adata, path=path, basis=basis, pseudotime_key=pseudotime_key)
                path_dict[basis] = path
            except:
                logger.info(f"Failed to compute shortest path for {basis}")
                continue

        time_dict[truetime_key] = adata.obs[truetime_key]
        pseudotime_result = compute_correlation(time_dict, corr_types=corr_types, groundtruth_key=truetime_key)
        pseudotime_result.columns = [f'{col}(pt)' for col in pseudotime_result.columns]
        results_df['Pseudotime'] = pseudotime_result.drop(truetime_key, inplace=False)
        results_full['Pseudotime'] = {
            'path': path_dict,
            'pseudotime': time_dict,
            'correlation': pseudotime_result
        }

    
    # Global distance correlation
    if 'cell_distance_corr' in eval_metrics:
        logger.info("Computing cell distance correlation")
        distance_result = pairwise_distance(adata, keys = keys, metric=dist_metric)           
        corr_result = compute_correlation(distance_result, corr_types=corr_types, groundtruth_key=groundtruth_key)
        results_df['cell_distance_corr'] = corr_result
        results_df['cell_distance_corr'].columns = [f'{col}(cd)' for col in corr_result.columns]
        results_full['cell_distance_corr'] = {
            'distance': distance_result,
            'correlation': corr_result
        }

    # Local vs distal correlation
    if 'local_distal_corr' in eval_metrics:
        logger.info("Computing local vs distal correlation")
        local_cor = {}
        distal_cor = {}
        corr_method = 'spearmanr' # Default to spearmanr
        for key in keys:
            local_cor[key], distal_cor[key] = local_vs_distal_corr(adata.obsm[groundtruth_key], adata.obsm[key], method=corr_method, local_percentile=local_percentile, distal_percentile=distal_percentile)

        local_cor_df = pd.DataFrame(local_cor, index = [f'Local {corr_method}']).T
        distal_cor_df = pd.DataFrame(distal_cor, index = [f'Distal {corr_method}']).T
        local_distal_corr_df = pd.concat([local_cor_df, distal_cor_df], axis=1)
        results_df['local_distal_corr'] = local_distal_corr_df
        results_full['local_distal_corr'] = local_distal_corr_df

    # Trustworthiness
    if 'trustworthiness' in eval_metrics:
        logger.info("Computing trustworthiness")
        trustworthiness_scores, trustworthiness_stats = compute_trustworthiness(adata, embedding_keys = keys, groundtruth=groundtruth_key, metric=dist_metric, n_neighbors=trustworthiness_n_neighbors)
        results_df['trustworthiness'] = trustworthiness_stats
        results_full['trustworthiness'] = {
            'scores': trustworthiness_scores,
            'stats': trustworthiness_stats
        }
        
    # Cluster centroid distances correlation
    if 'state_distance_corr' in eval_metrics:
        logger.info("Computing cluster centroid distances correlation")
        cluster_centroid_distances = {}
        for key in keys:
            cluster_centroid_distances[key] = compute_centroid_distance(adata, key, state_key)
            
        corr_dist_result = compute_correlation(cluster_centroid_distances, corr_types=corr_types, groundtruth_key=groundtruth_key)
        corr_dist_result.columns = [f'{col}(sd)' for col in corr_dist_result.columns]
        results_df['state_distance_corr'] = corr_dist_result
        results_full['state_distance_corr'] = {
            'distance': cluster_centroid_distances,
            'correlation': corr_dist_result
        }

    if 'state_dispersion_corr' in eval_metrics:
        logger.info("Computing state dispersion correlation")
        state_dispersion = {}
        for key in keys:
            state_dispersion[key] = compute_dispersion_across_states(adata, basis = key, state_key=state_key, dispersion_metric=dispersion_metric)
            
        if groundtruth_dispersion is not None:
            state_dispersion['Groundtruth'] = groundtruth_dispersion

        corr_dispersion_result = compute_correlation(state_dispersion, corr_types=corr_types, groundtruth_key='Groundtruth' if groundtruth_dispersion is not None else ground_truth_dispersion_key)
        corr_dispersion_result.columns = [f'{col}(sv)' for col in corr_dispersion_result.columns]
        results_df['state_dispersion_corr'] = corr_dispersion_result
        results_full['state_dispersion_corr'] = {
            'dispersion': state_dispersion,
            'correlation': corr_dispersion_result
        }

    # Batch-to-State Distance Ratio for all latent embeddings
    if 'state_batch_distance_ratio' in eval_metrics:
        logger.info("Computing state-batch distance ratio")
        state_batch_distance_ratios = {}
        for key in keys:
            state_batch_distance_ratios[key] = compute_state_batch_distance_ratio(adata, basis=key, batch_key=batch_key, state_key=state_key, metric='cosine')

        state_batch_distance_ratio_df = pd.DataFrame(state_batch_distance_ratios, index=[f'State-Batch Distance Ratio']).T
        state_batch_distance_ratio_df = np.log10(state_batch_distance_ratio_df)
        state_batch_distance_ratio_df.columns = [f'State-Batch Distance Ratio (log10)']
        # Set groundtruth to Nan
        if groundtruth_key in state_batch_distance_ratio_df.index:
            state_batch_distance_ratio_df.loc[groundtruth_key] = np.nan
        
        results_df['state_batch_distance_ratio'] = state_batch_distance_ratio_df
        results_full['state_batch_distance_ratio'] = state_batch_distance_ratio_df
    
    if save_dir is not None and file_suffix is not None:
        for key, result in results_df.items():
            result.to_csv(save_dir / f"{key}_{file_suffix}.csv")
    
    combined_results_df = pd.concat(results_df, axis=1)

    colname_mapping = {
        'cell_distance_corr': 'Cell distance correlation',
        'local_distal_corr': 'Cell distance correlation',
        'trustworthiness': 'Trustworthiness',
        'state_distance_corr': 'State distance',
        'state_dispersion_corr': 'Dispersion',
        'state_batch_distance_ratio': 'State/batch',
        'Average Trustworthiness': 'Mean',
        'Trustworthiness Decay (100N)': 'Decay',
        'State-Batch Distance Ratio (log10)': 'Distance ratio (log10)',
    }
    combined_results_df = combined_results_df.rename(columns=colname_mapping)

    if return_type == 'full':
        return combined_results_df, results_full
    else :
        return combined_results_df
    

def simplify_geometry_benchmark_table(df):
    # Simplify the dataframe by computing average for each metric
    if "Cell distance correlation" in df.columns.get_level_values(0):
        df[("Geometric metrics", "Cell distance correlation")] = df["Cell distance correlation"][
            ["pearsonr(cd)", "spearmanr(cd)", "kendalltau(cd)"]
        ].mean(axis=1)

        df.drop(
            columns=[
                ("Cell distance correlation", "pearsonr(cd)"),
                ("Cell distance correlation", "spearmanr(cd)"),
                ("Cell distance correlation", "kendalltau(cd)"),
                ("Cell distance correlation", "Local spearmanr"),
                ("Cell distance correlation", "Distal spearmanr")
            ],
            inplace=True
        )

    if "Trustworthiness" in df.columns.get_level_values(0):
        df[("Geometric metrics", "Trustworthiness")] = df["Trustworthiness"][["Mean"]]
        df.drop(
            columns=[
                ("Trustworthiness", "Decay"),
                ("Trustworthiness", "Mean")
            ],
            inplace=True
        )

    if "State distance" in df.columns.get_level_values(0):
        df[("Geometric metrics", "State distance correlation")] = df["State distance"][
            ["pearsonr(sd)", "spearmanr(sd)", "kendalltau(sd)"]
        ].mean(axis=1)
        df.drop(
            columns=[
                ("State distance", "pearsonr(sd)"),
                ("State distance", "spearmanr(sd)"),
                ("State distance", "kendalltau(sd)"),
            ],
            inplace=True
        )

    if "Dispersion" in df.columns.get_level_values(0):
        df[("Geometric metrics", "State dispersion correlation")] = df["Dispersion"][
            ["pearsonr(sv)", "spearmanr(sv)", "kendalltau(sv)"]
        ].mean(axis=1)
        df.drop(
            columns=[
                ("Dispersion", "pearsonr(sv)"),
                ("Dispersion", "spearmanr(sv)"),
                ("Dispersion", "kendalltau(sv)"),
            ],
            inplace=True
        )
    return df


# Convert benchmark table to scores
def benchmark_stats_to_score(df, fillna=None, min_max_scale=True, one_minus=False, aggregate_score=False, aggregate_score_name = ('Aggregate', 'Score'), name_exact=False, rank=False, rank_col=None):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    df = df.copy()
    if fillna is not None:
        df = df.fillna(fillna)

    if min_max_scale:
        df = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            columns=df.columns,
            index=df.index,
        )
        if name_exact:
            df.columns = pd.MultiIndex.from_tuples([(col[0], f"{col[1]}(min-max)") for col in df.columns])

    if one_minus:
        df = 1 - df
        if name_exact:
            df.columns = pd.MultiIndex.from_tuples([(col[0], f"1-{col[1]}") for col in df.columns])

    if aggregate_score:
        aggregate_df = pd.DataFrame(
            df.mean(axis=1),
            columns=pd.MultiIndex.from_tuples([aggregate_score_name]),
        )
        df = pd.concat([df, aggregate_df], axis=1)

    if rank:
        if rank_col is None:
            raise ValueError("rank_col must be specified when rank=True.")
        # Reorder the rows based on the aggregate score
        df = df.sort_values(by=rank_col, ascending=False)

    return df




# Recompute nmi and ari using the approach described in paper, with resolution range from 0.1 to 1.0 step 0.1
def compute_nmi_ari(adata, emb_key, label_key, resolution_range = np.arange(0.1, 1.1, 0.1), n_neighbors=30, metric='euclidean', verbose=True):
    import scanpy as sc
    import scib
    if verbose:
        logger.setLevel(logging.INFO)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=emb_key, metric=metric)
    cluster_key = f'leiden_{emb_key}'
    nmi_vals = []
    ari_vals = []
    for resolution in resolution_range:
        logger.info(f"Computing NMI and ARI for resolution {resolution}")
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
        nmi_vals.append(scib.metrics.nmi(adata, cluster_key, label_key))
        ari_vals.append(scib.metrics.ari(adata, cluster_key, label_key))
    
    return nmi_vals, ari_vals



def benchmark_nmi_ari(adata, emb_keys, label_key='cell_type', resolution_range = np.arange(0.1, 1.1, 0.1), n_neighbors=30, metric='euclidean', verbose=True):
    import pandas as pd
    if verbose:
        logger.setLevel(logging.INFO)
    nmi_vals = {}
    ari_vals = {}
    for key in emb_keys:
        logger.info(f"Computing NMI and ARI for {key}")
        nmi_vals[key], ari_vals[key] = compute_nmi_ari(adata, key, label_key, resolution_range=resolution_range, n_neighbors=n_neighbors, metric=metric, verbose=verbose)
    
    nmi_df = pd.DataFrame(nmi_vals, index=resolution_range)
    ari_df = pd.DataFrame(ari_vals, index=resolution_range)

    return nmi_df, ari_df



def probe_dict_to_df(results: dict,
                     label: str) -> pd.DataFrame:
    """
    Turn one probe’s nested-dict results into a DataFrame with a 2-level
    MultiIndex on the columns.

    Parameters
    ----------
    results     nested dict from your probe (linear, knn, …)
    label       outer-level column label (e.g. "Linear", "KNN")
    metric_map  dict mapping target_name ➜ metric_to_extract
                e.g. {"time": "r2", "leiden_no_noise": "accuracy", "batch": "accuracy"}
    """
    import pandas as pd
    cols = []
    for target in results.keys():
        # grab the Series (a column from the per-target DataFrame)
        if 'r2' in results[target].columns: 
            metric = 'r2'
        elif 'accuracy' in results[target].columns:
            metric = 'accuracy'
        elif 'error' in results[target].columns:
            metric = 'error'
        else:
            raise ValueError(f"Unknown metric in results for target {target}. Expected 'r2' or 'accuracy'.")
    
        s = results[target][metric].rename(
            f"{target}\n{metric}" 
        )
        cols.append(s)

    df = pd.concat(cols, axis=1)
    df.columns = pd.MultiIndex.from_product([[label], df.columns])
    return df





# ------------------------------------------------------------------ #
#  Utility: keep Aggregate-score blocks at the end
# ------------------------------------------------------------------ #
def move_aggregate_last(df: pd.DataFrame,
                        outer_name: str = "Aggregate score") -> pd.DataFrame:
    outer = df.columns.get_level_values(0)
    is_agg = outer == outer_name
    new_cols = list(df.columns[~is_agg]) + list(df.columns[is_agg])
    return df[new_cols]


# ------------------------------------------------------------------ #
#  1. SCIB benchmark
# ------------------------------------------------------------------ #
def run_scib_benchmark(adata,
                       *,
                       embedding_keys: Sequence[str],
                       batch_key: str,
                       state_key: str,
                       n_jobs: int = 6,
                       rank: bool = True,
                       save_table: Optional[Path] = None,
                       plot: bool = False,
                       plot_kw: Optional[dict] = None):
    from scib_metrics.benchmark import (
        Benchmarker, BioConservation, BatchCorrection,
    )

    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=state_key,
        embedding_obsm_keys=list(embedding_keys),
        n_jobs=n_jobs,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
    )
    bm.benchmark()
    scib_scores = bm.get_results(min_max_scale=False)

    # ── convert "Metric Type" row ➜ inner level of a MultiIndex column
    metric_type = scib_scores.loc["Metric Type"]
    scib_scores = scib_scores.drop("Metric Type")
    scib_scores.columns = pd.MultiIndex.from_tuples(
        [(metric_type[col], col) for col in scib_scores.columns]
    )

    # ── score + ranking (keep raw metrics: min_max_scale=False, one_minus=False)
    scib_scores = benchmark_stats_to_score(
        scib_scores,
        min_max_scale=False,
        one_minus=False,
        aggregate_score=False,
        rank=rank,
        rank_col=("Aggregate score", "Total"),
        name_exact=False,
    )

    if plot:
        plot_benchmark_table(
            scib_scores,
            agg_name="Aggregate score",
            **(plot_kw or {})
        )
        if save_table:
            plot_benchmark_table(
                scib_scores,
                save_path=save_table,
                agg_name="Aggregate score",
                **(plot_kw or {})
            )

    return scib_scores


# ------------------------------------------------------------------ #
#  2. Probe (linear + k-NN) benchmark
# ------------------------------------------------------------------ #
def run_probe_benchmark(adata,
                        *,
                        embedding_keys: Sequence[str],
                        state_key = "state",
                        batch_key = "batch",
                        ignore_values=("unannotated", "nan", "NaN", np.nan, "NA"),
                        rank: bool = True,
                        save_table: Optional[Path] = None,
                        plot: bool = False,
                        plot_kw: Optional[dict] = None):
    from concord.benchmarking import (
        LinearProbeEvaluator, KNNProbeEvaluator, probe_dict_to_df
    )

    # ── 2.1 run linear probe
    key_name_mapping = {
        "state": state_key,
        "batch": batch_key,
    }
    linear_res = {}
    for key in key_name_mapping.keys():
        logger.info(f"Running linear probe for {key} with keys {embedding_keys}")
        evaluator = LinearProbeEvaluator(
            adata, embedding_keys, key_name_mapping[key],
            task="auto", epochs=20, ignore_values=ignore_values,
            device="cpu", return_preds=False
        )
        linear_res[key] = evaluator.run()
        # invert batch accuracy by 1-acc
        if key == 'batch':
            linear_res[key]["error"] = 1 - linear_res[key]["accuracy"]
            linear_res[key].drop(columns=["accuracy"], inplace=True)

    # ── 2.2 run k-NN probe
    knn_res = {}
    for key in key_name_mapping.keys():
        logger.info(f"Running k-NN probe for {key} with keys {embedding_keys}")
        knn_eval = KNNProbeEvaluator(
            adata, embedding_keys, key_name_mapping[key], ignore_values=ignore_values, k=30
        )
        knn_res[key] = knn_eval.run()
        # invert batch accuracy by 1-acc
        if key == 'batch':
            knn_res[key]["error"] = 1 - knn_res[key]["accuracy"]
            knn_res[key].drop(columns=["accuracy"], inplace=True)

    # ── 2.3 collect into one DataFrame
    linear_df = probe_dict_to_df(linear_res, "Linear")
    knn_df    = probe_dict_to_df(knn_res,    "KNN")
    probe_df  = pd.concat([linear_df, knn_df], axis=1).sort_index(axis=1, level=0)

    # ── 2.4 score + rank
    probe_scores = benchmark_stats_to_score(
        probe_df,
        min_max_scale=False,
        one_minus=False,
        aggregate_score=True,
        aggregate_score_name=("Probe", "Score"),
        rank=rank,
        rank_col=("Probe", "Score"),
        name_exact=False,
    )
    probe_scores.index.name = "Method"

    if plot:
        plot_benchmark_table(
            probe_scores,
            agg_name="Probe",
            **(plot_kw or {})
        )
        if save_table:
            plot_benchmark_table(
                probe_scores,
                save_path=save_table,
                agg_name="Probe",
                **(plot_kw or {})
            )

    return probe_scores


# ------------------------------------------------------------------ #
#  3. Persistent-homology / topology benchmark
# ------------------------------------------------------------------ #
def run_topology_benchmark(adata,
                           *,
                           embedding_keys: Sequence[str],
                           save_dir: Path,
                           file_suffix: str,
                           homology_dimensions=(0, 1, 2),
                           expected_betti_numbers=(0, 0, 0),
                           rank: bool = True,
                           plot: bool = False,
                           plot_kw: Optional[dict] = None):
    diagrams = {}
    for key in embedding_keys:
        logger.info(f"Computing persistent homology for {key}")
        diagrams[key] = compute_persistent_homology(
            adata, key=key, homology_dimensions=homology_dimensions
        )

    with (save_dir / f"topology_diagrams_{file_suffix}.pkl").open("wb") as f:
        pickle.dump(diagrams, f)
    logger.info(f"Saved persistent homology diagrams to {save_dir / f'topology_diagrams_{file_suffix}.pkl'}")

    topo_res = benchmark_topology(
        diagrams, expected_betti_numbers=expected_betti_numbers,
        save_dir=save_dir, file_suffix=file_suffix
    )
    topo_df = topo_res["combined_metrics"]

    # cap very large distances
    topo_df[("Betti number", "L1 distance")] = topo_df[
        ("Betti number", "L1 distance")
    ].clip(upper=5)

    # score + rank
    topo_scores = benchmark_stats_to_score(
        topo_df,
        min_max_scale=True,
        one_minus=True,
        aggregate_score=True,
        aggregate_score_name=("Topology", "Score"),
        rank=rank,
        rank_col=("Topology", "Score"),
    )

    if plot:
        plot_benchmark_table(
            topo_scores,
            agg_name="Topology",
            **(plot_kw or {})
        )

    return topo_scores


# ------------------------------------------------------------------ #
#  4. Geometry benchmark
# ------------------------------------------------------------------ #
def run_geometry_benchmark(adata,
                           *,
                           embedding_keys: Sequence[str],
                           groundtruth_key: str,
                           state_key: str,
                           batch_key: str,
                           dist_metric: str = "cosine",
                           corr_types=("pearsonr", "spearmanr", "kendalltau"),
                           geometry_metrics=("cell_distance_corr", "trustworthiness",
                                             "state_dispersion_corr"),
                           rank: bool = True,
                           plot: bool = False,
                           plot_kw: Optional[dict] = None,
                           save_dir: Optional[Path] = None,
                           file_suffix: str = ""):
    bm_keys = [groundtruth_key, "wt_noise"] + list(embedding_keys)
    geom_df, geom_full = benchmark_geometry(
        adata,
        keys=bm_keys,
        eval_metrics=geometry_metrics,
        dist_metric=dist_metric,
        corr_types=corr_types,
        groundtruth_key=groundtruth_key,
        state_key=state_key,
        batch_key=batch_key,
        dispersion_metric="var",
        return_type="full",
        pseudotime_k=30,
        truetime_key="time",
        save_dir=save_dir,
        file_suffix=file_suffix,
    )

    # Save full results if save_dir is provided
    if save_dir is not None:
        geom_full_path = save_dir / f"geometry_results_{file_suffix}.pkl"
        with geom_full_path.open("wb") as f:
            pickle.dump(geom_full, f)
        logger.info(f"Saved full geometry benchmark results to {geom_full_path}")

    # drop the ground-truth rows
    geom_df = geom_df.drop(index=[groundtruth_key, "wt_noise"])

    # score + rank
    geom_scores = benchmark_stats_to_score(
        geom_df,
        fillna=0,
        min_max_scale=False,
        one_minus=False,
        aggregate_score=True,
        aggregate_score_name=("Geometry", "Score"),
        rank=rank,
        rank_col=("Geometry", "Score"),
    )

    if plot:
        plot_benchmark_table(
            geom_scores,
            agg_name="Geometry",
            **(plot_kw or {})
        )

    return geom_scores


# ------------------------------------------------------------------ #
#  5. Master pipeline
# ------------------------------------------------------------------ #
def run_benchmark_pipeline(
        adata,
        *,
        embedding_keys: Sequence[str],
        state_key: str,
        batch_key: str,
        groundtruth_key: Optional[dict[str, str]] = None,
        save_dir: Path = Path("benchmark_results"),
        file_suffix: str = "",
        run: Sequence[Literal["scib", "probe", "topology", "geometry"]] = (
            "scib", "probe", "topology", "geometry"),
        plot_individual: bool = True,
        combine_plots: bool = True,
        table_plot_kw: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """
    Run selected benchmarking blocks and return a dict with their score tables
    plus a combined table (key = "combined").

    Parameters
    ----------
    run               : which blocks to execute
    plot_individual   : show / save each block's table
    combine_plots     : plot the final merged table
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    table_plot_kw = table_plot_kw or dict(
        pal="PRGn", pal_agg="RdYlBu_r", cmap_method="minmax", dpi=300)

    results: dict[str, pd.DataFrame] = {}

    if "scib" in run:
        logger.info("Running SCIB benchmark")
        results["scib"] = run_scib_benchmark(
            adata,
            embedding_keys=embedding_keys,
            batch_key=batch_key,
            state_key=state_key,
            save_table=save_dir / f"scib_results_{file_suffix}.pdf",
            plot=plot_individual,
            plot_kw=table_plot_kw
        )

    if "probe" in run:
        logger.info("Running Probe benchmark")
        results["probe"] = run_probe_benchmark(
            adata,
            embedding_keys=embedding_keys,
            state_key=state_key,
            batch_key=batch_key,
            ignore_values=("unannotated", "nan", "NaN", np.nan, "NA"),
            save_table=save_dir / f"probe_results_{file_suffix}.pdf",
            plot=plot_individual,
            plot_kw=table_plot_kw
        )

    if "topology" in run:
        logger.info("Running Topology benchmark")
        results["topology"] = run_topology_benchmark(
            adata,
            embedding_keys=embedding_keys,
            save_dir=save_dir,
            file_suffix=file_suffix,
            plot=plot_individual,
            plot_kw=table_plot_kw
        )

    if "geometry" in run:
        logger.info("Running Geometry benchmark")
        if groundtruth_key is None or groundtruth_key not in adata.obsm:
            logger.error(
                "groundtruth_key must be provided when running geometry benchmark. skipping."
            )
        else:
            results["geometry"] = run_geometry_benchmark(
                    adata,
                    embedding_keys=embedding_keys,
                    groundtruth_key=groundtruth_key,
                    state_key=state_key,
                    batch_key=batch_key,
                    save_dir=save_dir,
                    file_suffix=file_suffix,
                    plot=plot_individual,
                    plot_kw=table_plot_kw
                )

    # ------------------------------------------------------------------
    #  Combine selected results
    # ------------------------------------------------------------------
    to_concat = []
    for block in ("geometry", "topology", "scib", "probe"):
        if block in results:
            df = results[block].copy()
            if block == "geometry":
                df[("Aggregate score", "Geometry")] = df[("Geometry", "Score")]
                df.drop(columns=[("Geometry", "Score")], inplace=True)
            elif block == "topology":
                df[("Aggregate score", "Topology")] = df[("Topology", "Score")]
                df.drop(columns=[("Topology", "Score")], inplace=True)
            elif block == "probe":
                df[("Aggregate score", "Probe")] = df[("Probe", "Score")]
                df.drop(columns=[("Probe", "Score")], inplace=True)
            elif block == "scib":
                df.drop(columns=[("Aggregate score", "Total")], inplace=True)
            to_concat.append(df)

    combined_df = pd.concat(to_concat, axis=1)
    combined_df = move_aggregate_last(combined_df, "Aggregate score")

    # Average aggregate
    agg_cols = combined_df.columns[combined_df.columns.get_level_values(0)
                                    == "Aggregate score"]
    combined_df[("Aggregate score", "Average")] = combined_df[agg_cols].mean(axis=1)
    combined_df = combined_df.sort_values(
        by=("Aggregate score", "Average"), ascending=False)

    results["combined"] = combined_df

    if combine_plots:
        plot_benchmark_table(
            combined_df,
            save_path=save_dir / f"combined_res_{file_suffix}.pdf",
            agg_name="Aggregate score",
            figsize=(max(8, 1.5 * len(combined_df.columns)), 7),
            **table_plot_kw
        )

    return results




