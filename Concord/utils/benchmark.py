
import pandas as pd
from .timer import Timer
import logging
from .. import logger


def count_total_runs(param_grid):
    total_runs = 0
    for key, values in param_grid.items():
        if isinstance(values, list):
            total_runs += len(values)
    return total_runs

def run_hyperparameter_tests(adata, base_params, param_grid, output_key = "X_concord", return_decoded=False, trace_memory=False, trace_gpu_memory=False, save_dir="./"):
    import time
    import json
    from pathlib import Path
    from copy import deepcopy
    from .timer import Timer
    from .anndata_utils import save_obsm_to_hdf5
    import tracemalloc

    total_runs = count_total_runs(param_grid)
    logger.info(f"Total number of runs: {total_runs}")

    log_df = pd.DataFrame(columns=["param_name", "value", "time_minutes", "peak_memory_MB", "peak_gpu_memory_MB"])

    if trace_gpu_memory:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU setup
        except ImportError:
            raise ImportError("pynvml is not available.")

    for param_name, values in param_grid.items():
        for value in values:
            # Create a deep copy of base_params to avoid modifying the original dictionary
            params_copy = deepcopy(base_params)
            
            with Timer() as timer:

                if trace_memory:
                    tracemalloc.start()

                if trace_gpu_memory:
                    # Get initial GPU memory usage
                    initial_gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used

                logger.info(f"Encoding adata with {param_name}={value}.")
                params_copy[param_name] = value

                # Initialize Concord model with updated parameters
                from ..concord import Concord
                ccd = Concord(adata=adata, **params_copy)

                # Define the output key and file suffix including param_name and value
                output_key_final = f"{output_key}_{param_name}_{str(value).replace(' ', '')}"
                file_suffix = f"{param_name}_{str(value).replace(' ', '')}_{time.strftime('%b%d-%H%M')}"


                # Encode adata and store the results in adata.obsm
                ccd.encode_adata(input_layer_key="X_log1p", output_key=output_key_final, return_decoded=return_decoded)
                adata.obsm[output_key_final] = ccd.adata.obsm[output_key_final]

                if trace_memory:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_memory_MB = peak / (1024 * 1024)  # Convert to MB
                else:
                    peak_memory_MB = None

                if trace_gpu_memory:
                    # Get final GPU memory usage and calculate peak usage
                    final_gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used
                    peak_gpu_memory_MB = (final_gpu_memory - initial_gpu_memory) / (1024 * 1024)  # Convert to MB
                else:
                    peak_gpu_memory_MB = None


                # Save the parameter settings
                config_filename = Path(save_dir) / f"config_{file_suffix}.json"
                with open(config_filename, 'w') as f:
                    json.dump(ccd.config.to_dict(), f, indent=4)

            time_minutes = timer.interval / 60
            logger.info(f"Took {time_minutes:.2f} minutes to encode adata with {param_name}={value}, result saved to adata.obsm['{output_key}'], config saved to {config_filename}.")

            log_data = pd.DataFrame([{
                "param_name": param_name,
                "value": value,
                "time_minutes": time_minutes,
                "peak_memory_MB": peak_memory_MB,
                "peak_gpu_memory_MB": peak_gpu_memory_MB
            }])
            log_df = pd.concat([log_df, log_data], ignore_index=True)

    if trace_gpu_memory:
        pynvml.nvmlShutdown()

    # Save the entire adata.obsm to file after all tests
    obsm_filename =  Path(save_dir) / f"final_obsm_{file_suffix}.h5"
    save_obsm_to_hdf5(adata, obsm_filename)

    log_filename = Path(save_dir) / f"hyperparameter_test_log_{file_suffix}.csv"
    log_df.to_csv(log_filename, index=False)

    return adata



def run_scanorama(adata, batch_key="batch", output_key="Scanorama", return_corrected=False):
    import scanorama
    import numpy as np

    batch_cats = adata.obs[batch_key].cat.categories
    adata_list = [adata[adata.obs[batch_key] == b].copy() for b in batch_cats]

    scanorama.integrate_scanpy(adata_list)
    adata.obsm[output_key] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))

    if return_corrected:
        corrected = scanorama.correct_scanpy(adata_list)
        adata.layers[output_key + "_corrected"] = np.zeros(adata.shape)
    for i, b in enumerate(batch_cats):
        adata.obsm[output_key][adata.obs[batch_key] == b] = adata_list[i].obsm["X_scanorama"]
        if return_corrected:
            adata.layers[output_key + "_corrected"][adata.obs[batch_key] == b] = corrected[i].X.toarray()



def run_liger(adata, batch_key="batch", count_layer="counts", output_key="LIGER", k=30, return_corrected=False):
    import numpy as np
    import pyliger
    from scipy.sparse import csr_matrix

    bdata = adata.copy()
    # Ensure batch_key is a categorical variable
    if not bdata.obs[batch_key].dtype.name == "category":
        bdata.obs[batch_key] = bdata.obs[batch_key].astype("category")
    batch_cats = bdata.obs[batch_key].cat.categories

    # Set the count layer as the primary data for normalization in Pyliger    
    bdata.X = bdata.layers[count_layer]
    # Convert to csr matrix if not
    if not isinstance(bdata.X, csr_matrix):
        bdata.X = csr_matrix(bdata.X)
    
    # Create a list of adata objects, one per batch
    adata_list = [bdata[bdata.obs[batch_key] == b].copy() for b in batch_cats]
    for i, ad in enumerate(adata_list):
        ad.uns["sample_name"] = batch_cats[i]
        ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)  # Ensures same genes are used in each adata

    # Create a LIGER object from the list of adata per batch
    liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
    liger_data.var_genes = bdata.var_names  # Set genes for LIGER data consistency

    # Run LIGER integration steps
    pyliger.normalize(liger_data)
    pyliger.scale_not_center(liger_data)
    pyliger.optimize_ALS(liger_data, k=k)
    pyliger.quantile_norm(liger_data)


    # Initialize the obsm field for the integrated data
    adata.obsm[output_key] = np.zeros((adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
    
    # Populate the integrated embeddings back into the main AnnData object
    for i, b in enumerate(batch_cats):
        adata.obsm[output_key][adata.obs[batch_key] == b] = liger_data.adata_list[i].obsm["H_norm"]

    if return_corrected:
        corrected_expression = np.zeros(adata.shape)
        for i, b in enumerate(batch_cats):
            H = liger_data.adata_list[i].obsm["H_norm"]  # Latent representation (cells x factors)
            W = liger_data.W  # Gene loadings (genes x factors)
            corrected_expression[adata.obs[batch_key] == b] = H @ W.T

        adata.layers[output_key + "_corrected"] = corrected_expression
    

def run_harmony(adata, batch_key="batch", output_key="Harmony", input_key="X_pca"):
    from harmony import harmonize
    if input_key not in adata.obsm:
        raise ValueError(f"Input key '{input_key}' not found in adata.obsm")
    
    adata.obsm[output_key] = harmonize(adata.obsm[input_key], adata.obs, batch_key=batch_key)



def run_scvi(adata, layer="counts", batch_key="batch", gene_likelihood="nb", n_layers=2, n_latent=30, output_key="scVI", return_model=False, return_corrected=False, transform_batch=None):
    import scvi
    # Set up the AnnData object for SCVI
    scvi.model.SCVI.setup_anndata(adata, layer=layer, batch_key=batch_key)
    
    # Initialize and train the SCVI model
    vae = scvi.model.SCVI(adata, gene_likelihood=gene_likelihood, n_layers=n_layers, n_latent=n_latent)
    vae.train()
    
    # Store the latent representation in the specified obsm key
    adata.obsm[output_key] = vae.get_latent_representation()

    if return_corrected:
        corrected_expression = vae.get_normalized_expression(transform_batch=transform_batch)
        adata.layers[output_key + "_corrected" + ("" if transform_batch is None else f"_{transform_batch}")] = corrected_expression
    
    if return_model:
        return vae
    

def run_scanvi(adata, scvi_model=None, layer="counts", batch_key="batch", labels_key="cell_type", unlabeled_category="Unknown", output_key="scANVI", 
               gene_likelihood="nb", n_layers=2, n_latent=30, return_corrected=False, transform_batch=None):
    import scvi
    # Train SCVI model if not supplied
    if scvi_model is None:
        scvi_model = run_scvi(adata, layer=layer, batch_key=batch_key, gene_likelihood=gene_likelihood,
                              n_layers=n_layers, n_latent=n_latent, output_key="scVI")
    
    # Set up and train the SCANVI model
    lvae = scvi.model.SCANVI.from_scvi_model(scvi_model, adata=adata, labels_key=labels_key, unlabeled_category=unlabeled_category)
    lvae.train(max_epochs=20, n_samples_per_label=100)
    
    if return_corrected:
        corrected_expression = lvae.get_normalized_expression(transform_batch=transform_batch)
        adata.layers[output_key + "_corrected" + ("" if transform_batch is None else f"_{transform_batch}")] = corrected_expression
    # Store the SCANVI latent representation in the specified obsm key
    adata.obsm[output_key] = lvae.get_latent_representation()
    



def benchmark_topology(adata, keys, homology_dimensions=[0,1,2], expected_betti_numbers=[0,0,0], n_bins=100):
    import pandas as pd
    from .tda import compute_persistent_homology, compute_betti_statistics, summarize_betti_statistics
    dg_list = {}
    for key in keys:
        logger.info(f"Computing persistent homology for {key}")
        dg_list[key] =  compute_persistent_homology(adata, key=key, homology_dimensions=homology_dimensions)

    expected_betti_numbers = [4,0,0]
    betti_stats = {}    
    # Compute betti stats for all keys
    for key in dg_list.keys():
        betti_stats[key] = compute_betti_statistics(diagram=dg_list[key], expected_betti_numbers=expected_betti_numbers, n_bins=n_bins)

    betti_stats_pivot, distance_metrics_df = summarize_betti_statistics(betti_stats)

    entropy_columns = betti_stats_pivot.loc[:, pd.IndexSlice[:, 'Entropy']]
    average_entropy = entropy_columns.mean(axis=1)
    variance_columns = betti_stats_pivot.loc[:, pd.IndexSlice[:, 'Variance']]
    average_variance = variance_columns.mean(axis=1)
    final_metrics = pd.concat([average_entropy, average_variance], axis=1)
    final_metrics.columns = ['Entropy', 'Variance']
    final_metrics['L1 distance'] = distance_metrics_df['L1 Distance']
    
    return final_metrics, dg_list, betti_stats_pivot, distance_metrics_df



def benchmark_geometry(adata, keys, 
                       eval_metrics=['cell_distance_corr', 'local_distal_corr', 'trustworthiness', 'cluster_distance_corr', 'batch_state_distance_ratio'],
                       dist_metric='cosine', 
                       groundtruth_key = 'PCA_no_noise', 
                       state_key = 'cluster',
                       corr_types = ['pearsonr', 'spearmanr', 'kendalltau'], 
                       trustworthiness_n_neighbors = np.arange(10, 101, 10),
                       return_type='dataframe',
                       verbose=True,
                       save_dir=None, 
                       file_suffix=None):
    import pandas as pd
    from .geometry import pairwise_distance, distance_correlation, local_vs_distal_corr, compute_trustworthiness, compute_centroid_distance, compute_batch_state_distance_ratio

    results = {}
    if verbose:
        logger.setLevel(logging.INFO)

    # Global distance correlation
    if 'cell_distance_corr' in eval_metrics:
        logger.info("Computing cell distance correlation")
        distance_result = pairwise_distance(adata, keys = keys, metric=dist_metric)
        corr_result = distance_correlation(distance_result, corr_types=corr_types, groundtruth_key=groundtruth_key)
        corr_result.columns = [f'cell_{col}' for col in corr_result.columns]
        results['cell_distance_corr'] = corr_result

    # Local vs distal correlation
    if 'local_distal_corr' in eval_metrics:
        logger.info("Computing local vs distal correlation")
        local_spearman = {}
        distal_spearman = {}
        corr_method = 'spearman'
        for key in keys:
            local_spearman[key], distal_spearman[key] = local_vs_distal_corr(adata.obsm[groundtruth_key], adata.obsm[key], method=corr_method)

        local_spearman_df = pd.DataFrame(local_spearman, index = [f'Local {corr_method} correlation']).T
        distal_spearman_df = pd.DataFrame(distal_spearman, index = [f'Distal {corr_method} correlation']).T
        local_distal_corr_df = pd.concat([local_spearman_df, distal_spearman_df], axis=1)
        results['local_distal_corr'] = local_distal_corr_df

    # Trustworthiness
    if 'trustworthiness' in eval_metrics:
        logger.info("Computing trustworthiness")
        trustworthiness_scores, trustworthiness_stats = compute_trustworthiness(adata, embedding_keys = keys, groundtruth=groundtruth_key, metric=dist_metric, n_neighbors=trustworthiness_n_neighbors)
        results['trustworthiness'] = trustworthiness_stats
        
    # Cluster centroid distances correlation
    if 'cluster_distance_corr' in eval_metrics:
        logger.info("Computing cluster centroid distances correlation")
        cluster_centroid_distances = {}
        for key in keys:
            cluster_centroid_distances[key] = compute_centroid_distance(adata, key, state_key)
            
        corr_cluster_result = distance_correlation(cluster_centroid_distances, corr_types=corr_types, groundtruth_key=groundtruth_key)
        corr_cluster_result.columns = [f'{state_key}_{col}' for col in corr_cluster_result.columns]
        results['cluster_distance_corr'] = corr_cluster_result

    # Batch-to-State Distance Ratio for all latent embeddings
    if 'batch_state_distance_ratio' in eval_metrics:
        logger.info("Computing batch-state distance ratio")
        batch_state_distance_ratios = {}
        for key in keys:
            batch_state_distance_ratios[key] = compute_batch_state_distance_ratio(adata, latent_key=key, batch_key=batch_key, state_key=state_key, metric='cosine')

        batch_state_distance_ratio_df = pd.DataFrame(batch_state_distance_ratios, index=[f'Batch-State Distance Ratio']).T
        results['batch_state_distance_ratio'] = batch_state_distance_ratio_df
    
    if save_dir is not None and file_suffix is not None:
        for key, result in results.items():
            result.to_csv(save_dir / f"{key}_{file_suffix}.csv")
    
    # Combine the results into a single dataframe
    if return_type == 'dict':
        return results
    else :
        # Merge the results into a single dataframe using the same index
        combined_results = pd.concat(results, axis=1)
        return combined_results
    