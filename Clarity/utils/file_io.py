
import anndata as ad
import h5py
import numpy as np

def save_obsm_to_hdf5(adata, filename):
    """
    Save the .obsm attribute of an AnnData object to an HDF5 file.

    Parameters:
    adata (anndata.AnnData): The AnnData object containing the .obsm attribute.
    filename (str): The name of the HDF5 file to save the data to.
    """
    with h5py.File(filename, 'w') as f:
        for key, matrix in adata.obsm.items():
            f.create_dataset(key, data=matrix)


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
        # Assuming the .obsm data is stored under a group named 'obsm'
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

