
import pandas as pd
from .timer import Timer
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



def run_scanorama(adata, batch_key="batch", output_key="Scanorama"):
    import scanorama
    import numpy as np

    batch_cats = adata.obs[batch_key].cat.categories
    adata_list = [adata[adata.obs[batch_key] == b].copy() for b in batch_cats]

    scanorama.integrate_scanpy(adata_list)

    adata.obsm[output_key] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
    for i, b in enumerate(batch_cats):
        adata.obsm[output_key][adata.obs[batch_key] == b] = adata_list[i].obsm["X_scanorama"]



def run_liger(adata, batch_key="batch", count_layer="counts", output_key="LIGER", k=30):
    import numpy as np
    import pyliger
    from scipy.sparse import csr_matrix

    bdata = adata.copy()
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
    

def run_harmony(adata, batch_key="batch", output_key="Harmony", input_key="X_pca"):
    from harmony import harmonize
    if input_key not in adata.obsm:
        raise ValueError(f"Input key '{input_key}' not found in adata.obsm")
    
    adata.obsm[output_key] = harmonize(adata.obsm[input_key], adata.obs, batch_key=batch_key)



def run_scvi(adata, layer="counts", batch_key="batch", gene_likelihood="nb", n_layers=2, n_latent=30, output_key="scVI", return_model=False):
    import scvi
    # Set up the AnnData object for SCVI
    scvi.model.SCVI.setup_anndata(adata, layer=layer, batch_key=batch_key)
    
    # Initialize and train the SCVI model
    vae = scvi.model.SCVI(adata, gene_likelihood=gene_likelihood, n_layers=n_layers, n_latent=n_latent)
    vae.train()
    
    # Store the latent representation in the specified obsm key
    adata.obsm[output_key] = vae.get_latent_representation()
    
    if return_model:
        return vae
    

def run_scanvi(adata, scvi_model=None, layer="counts", batch_key="batch", labels_key="cell_type", unlabeled_category="Unknown", output_key="scANVI", 
               gene_likelihood="nb", n_layers=2, n_latent=30):
    import scvi
    # Train SCVI model if not supplied
    if scvi_model is None:
        scvi_model = run_scvi(adata, layer=layer, batch_key=batch_key, gene_likelihood=gene_likelihood,
                              n_layers=n_layers, n_latent=n_latent, output_key="scVI")
    
    # Set up and train the SCANVI model
    lvae = scvi.model.SCANVI.from_scvi_model(scvi_model, adata=adata, labels_key=labels_key, unlabeled_category=unlabeled_category)
    lvae.train(max_epochs=20, n_samples_per_label=100)
    
    # Store the SCANVI latent representation in the specified obsm key
    adata.obsm[output_key] = lvae.get_latent_representation()
    

