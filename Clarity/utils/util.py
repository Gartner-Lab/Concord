import random
import numpy as np
import torch
import logging
from pathlib import Path
import os
import glob
import scanpy as sc
import anndata as ad
import wandb
import pandas as pd
from typing import Optional

def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)



def update_wandb_params(params, project_name=None, reinit=True, start_method="thread"):
    if reinit:
        run = wandb.init(
            config=params,
            project=project_name,
            reinit=True,
            settings=wandb.Settings(start_method=start_method),
        )
    else:
        run = wandb.run

    config = wandb.config
    config.update(params)
    return config, run


def list_adata_files(folder_path, substring=None, extension='*'):
    """
    List all .h5ad files in a folder recursively that contain a specific substring in their names.

    Args:
        folder_path (str): The path to the folder to search.
        substring (str): The substring to filter filenames.

    Returns:
        list: A list of file paths that contain the substring.
    """
    # Use glob to find all .h5ad files recursively
    all_files = glob.glob(os.path.join(folder_path, '**', extension), recursive=True)

    # Filter files that contain the substring in their names
    if substring is not None:
        filtered_files = [f for f in all_files if substring in os.path.basename(f)]
    else:
        filtered_files = all_files

    return filtered_files


def read_and_concatenate_adata(adata_files, merge='unique', add_dataset_column=False):
    """
    Read all .h5ad files and concatenate them into a single AnnData object.

    Optionally append a column 'dataset' to each AnnData object containing values
    derived from the file name (stripping away the extension), so that after concatenation
    the column indicates the original dataset.

    Args:
        adata_files (list of str): List of file paths to .h5ad files.
        merge (str): How to merge AnnData objects. Options are 'unique', 'same', 'first', 'override'.
        add_dataset_column (bool): Whether to add a 'dataset' column indicating the source file.

    Returns:
        AnnData: The concatenated AnnData object.
    """
    adata_list = []

    for file in adata_files:
        adata = sc.read_h5ad(file)
        if add_dataset_column:
            dataset_name = os.path.splitext(os.path.basename(file))[0]
            adata.obs['dataset'] = dataset_name
        adata_list.append(adata)

    # Concatenate all AnnData objects into a single AnnData object
    adata_combined = ad.concat(adata_list, axis=0, join='outer', merge=merge)
    return adata_combined



def filter_and_copy_attributes(adata_target, adata_source):
    """
    Filter adata_target to have the same cells as adata_source and copy
    adata_source's obs and obsm to adata_target.

    Parameters:
    adata_target (anndata.AnnData): The original AnnData object to be filtered.
    adata_source (anndata.AnnData): The AnnData object with the desired cell set and attributes.

    Returns:
    adata_filtered (anndata.AnnData): The filtered AnnData object with updated obs and obsm.
    """
    # Ensure the cell names are consistent and take the intersection
    cells_to_keep = adata_target.obs_names.intersection(adata_source.obs_names)
    cells_to_keep = list(cells_to_keep)  # Convert to list

    # Filter adata_target to retain only the intersected cells
    adata_filtered = adata_target[cells_to_keep].copy()

    # Copy obs from adata_source to adata_filtered for the intersected cells
    adata_filtered.obs = adata_source.obs.loc[cells_to_keep].copy()

    # Copy obsm from adata_source to adata_filtered for the intersected cells
    for key in adata_source.obsm_keys():
        adata_filtered.obsm[key] = adata_source.obsm[key][adata_source.obs_names.isin(cells_to_keep), :]

    # Ensure the raw attribute is set and var index is consistent
    if adata_filtered.raw is not None:
        adata_filtered.raw.var.index = adata_filtered.var.index
    else:
        adata_filtered.raw = adata_filtered.copy()
        adata_filtered.raw.var.index = adata_filtered.var.index

    return adata_filtered


def ensure_categorical(adata: ad.AnnData, obs_key: Optional[str] = None, drop_unused: bool = True):
    """
    Ensure that a specified column in the AnnData object is of the category dtype.
    Optionally drop unused levels if the column is already a category dtype.

    Parameters:
    adata : ad.AnnData
        The AnnData object.
    obs_key : str, optional
        The key of the column to ensure as categorical.
    drop_unused : bool, optional
        Whether to drop unused levels in the categorical column.
    """
    if obs_key in adata.obs:
        if not isinstance(adata.obs[obs_key].dtype, pd.CategoricalDtype):
            adata.obs[obs_key] = adata.obs[obs_key].astype('category')
            print(f"Column '{obs_key}' is now of type: {adata.obs[obs_key].dtype}")
        else:
            print(f"Column '{obs_key}' is already of type: {adata.obs[obs_key].dtype}")
            if drop_unused:
                adata.obs[obs_key] = adata.obs[obs_key].cat.remove_unused_categories()
                print(f"Unused levels dropped for column '{obs_key}'.")
    else:
        print(f"Column '{obs_key}' does not exist in the AnnData object.")