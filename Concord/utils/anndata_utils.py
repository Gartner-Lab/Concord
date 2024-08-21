import glob
import os
from typing import Optional
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from .. import logger

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



# Backed mode does not work now, this function (https://anndata.readthedocs.io/en/latest/generated/anndata.experimental.concat_on_disk.html) also has limitation
def read_and_concatenate_adata(adata_files, merge='unique', add_dataset_column=False, output_file=None, use_backed=False):
    """
    Read all .h5ad files and concatenate them into a single AnnData object.

    Optionally append a column 'dataset' to each AnnData object containing values
    derived from the file name (stripping away the extension), so that after concatenation
    the column indicates the original dataset.

    Args:
        adata_files (list of str): List of file paths to .h5ad files.
        merge (str): How to merge AnnData objects. Options are 'unique', 'same', 'first', 'override'.
        add_dataset_column (bool): Whether to add a 'dataset' column indicating the source file.
        output_file (str): If provided, the concatenated AnnData object will be saved to this file.
        use_backed (bool): Whether to use backed mode for memory-efficient concatenation.

    Returns:
        AnnData: The concatenated AnnData object.
    """
    if use_backed:
        if output_file is None:
            raise ValueError("Output file must be provided when using backed mode.")

        # Start by writing the first file to the output
        first_file = adata_files[0]
        adata_combined = sc.read_h5ad(first_file)
        
        if add_dataset_column:
            dataset_name = os.path.splitext(os.path.basename(first_file))[0]
            adata_combined.obs['dataset'] = dataset_name
        
        adata_combined.write(output_file)  # Save the initial file as the starting point
        adata_combined = sc.read_h5ad(output_file, backed='r+')  # Open it in backed mode for appending

        # Process the remaining files
        for file in adata_files[1:]:
            logger.info("Reading and concatenating file: " + file)
            adata = sc.read_h5ad(file)  # Load the next file in memory
            if add_dataset_column:
                dataset_name = os.path.splitext(os.path.basename(file))[0]
                adata.obs['dataset'] = dataset_name
            
            # Append the new data to the combined AnnData object
            adata_combined = ad.concat([adata_combined, adata], axis=0, join='outer', merge=merge)
            
            # Write the updated combined AnnData object back to disk
            adata_combined.write(output_file)
            adata_combined = sc.read_h5ad(output_file, backed='r+')  # Reopen in backed mode for the next iteration

            # Free up memory
            del adata

        return sc.read_h5ad(output_file, backed='r')
    
    else:
        # Standard concatenation in memory for smaller datasets
        adata_combined = None

        for file in adata_files:
            adata = sc.read_h5ad(file)  # Load the AnnData object in memory
            
            if add_dataset_column:
                dataset_name = os.path.splitext(os.path.basename(file))[0]
                adata.obs['dataset'] = dataset_name
            
            if adata_combined is None:
                adata_combined = adata
            else:
                adata_combined = ad.concat([adata_combined, adata], axis=0, join='outer', merge=merge)
            
            # Immediately delete the loaded adata to free up memory
            del adata

        if output_file is not None:
            adata_combined.write(output_file)  # Save the final concatenated object to disk

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
            logger.info(f"Column '{obs_key}' is now of type: {adata.obs[obs_key].dtype}")
        else:
            logger.info(f"Column '{obs_key}' is already of type: {adata.obs[obs_key].dtype}")
            if drop_unused:
                adata.obs[obs_key] = adata.obs[obs_key].cat.remove_unused_categories()
                logger.info(f"Unused levels dropped for column '{obs_key}'.")
    else:
        logger.warning(f"Column '{obs_key}' does not exist in the AnnData object.")




def save_obsm_to_hdf5(adata, filename):
    """
    Save the .obsm attribute of an AnnData object to an HDF5 file.

    Parameters:
    adata (anndata.AnnData): The AnnData object containing the .obsm attribute.
    filename (str): The name of the HDF5 file to save the data to.
    """
    with h5py.File(filename, 'w') as f:
        obsm_group = f.create_group('obsm')
        for key, matrix in adata.obsm.items():
            obsm_group.create_dataset(key, data=matrix)


def load_obsm_from_hdf5(filename):
    """
    Load the .obsm attribute from an HDF5 file.

    Parameters:
    filename (str): The name of the HDF5 file to read the data from.

    Returns:
    dict: A dictionary containing the loaded .obsm data.
    """
    obsm = {}
    with h5py.File(filename, 'r') as f:
        obsm_group = f['obsm']
        for key in obsm_group.keys():
            obsm[key] = obsm_group[key][:]
    return obsm


def subset_adata_to_obsm_indices(adata, obsm):
    """
    Subset the AnnData object to only contain the indices in obsm.

    Parameters:
    adata (anndata.AnnData): The original AnnData object.
    obsm (dict): The dictionary containing the .obsm data.

    Returns:
    anndata.AnnData: The subsetted AnnData object.
    """
    # Find the common indices across all obsm arrays
    indices = np.arange(adata.n_obs)
    for key in obsm:
        if obsm[key].shape[0] < indices.shape[0]:
            indices = indices[:obsm[key].shape[0]]

    # Subset the AnnData object
    adata_subset = adata[indices, :].copy()
    adata_subset.obsm = obsm
    return adata_subset


