
from __future__ import annotations
import anndata as ad
import os  
import scanpy as sc 
import numpy as np
from typing import List, Optional
from .. import logger


def list_adata_files(folder_path, substring=None, extension='*.h5ad'):
    """
    List all `.h5ad` files in a directory (recursively) that match a given substring.

    Args:
        folder_path : str
            Path to the folder where `.h5ad` files are located.
        substring : str, optional
            A substring to filter filenames (default is None, meaning no filtering).
        extension : str, optional
            File extension to search for (default is "*.h5ad").

    Returns:
        list
            A list of file paths matching the criteria.
    """
    import glob
    import os
    # Use glob to find all files with the specified extension recursively
    all_files = glob.glob(os.path.join(folder_path, '**', extension), recursive=True)
    
    # Filter files that contain the substring in their names
    if substring is not None:
        filtered_files = [f for f in all_files if substring in os.path.basename(f)]
    else:
        filtered_files = all_files

    return filtered_files



# Backed mode does not work now, this function (https://anndata.readthedocs.io/en/latest/generated/anndata.experimental.concat_on_disk.html) also has limitation
def read_and_concatenate_adata(adata_files, merge='unique', add_dataset_col=False, dataset_col_name = 'dataset', output_file=None):
    """
    Read and concatenate multiple AnnData `.h5ad` files into a single AnnData object.

    Args:
        adata_files : list
            List of file paths to `.h5ad` files to be concatenated.
        merge : str, optional
            How to handle conflicting columns, e.g., 'unique' (default), 'first', etc.
        add_dataset_col : bool, optional
            Whether to add a new column in `adata.obs` identifying the source dataset.
        dataset_col_name : str, optional
            Name of the new column storing dataset names.
        output_file : str, optional
            Path to save the concatenated AnnData object. If None, the object is not saved.

    Returns:
        ad.AnnData
            The concatenated AnnData object.
    """
    import gc
    # Standard concatenation in memory for smaller datasets
    adata_combined = None

    for file in adata_files:
        logger.info(f"Loading file: {file}")
        adata = sc.read_h5ad(file)  # Load the AnnData object in memory
        
        if add_dataset_col:
            dataset_name = os.path.splitext(os.path.basename(file))[0]
            adata.obs[dataset_col_name] = dataset_name
        
        if adata_combined is None:
            adata_combined = adata
        else:
            adata_combined = ad.concat([adata_combined, adata], axis=0, join='outer', merge=merge)
        
        logger.info(f"Combined shape: {adata_combined.shape}")
        # Immediately delete the loaded adata to free up memory
        del adata
        gc.collect()

    if output_file is not None:
        adata_combined.write(output_file)  # Save the final concatenated object to disk

    return adata_combined



def filter_and_copy_attributes(adata_target, adata_source):
    """
    Filter `adata_target` to match the cells in `adata_source`, then copy `.obs` and `.obsm`.

    Args:
        adata_target : ad.AnnData
            The AnnData object to be filtered.
        adata_source : ad.AnnData
            The reference AnnData object containing the desired cells and attributes.

    Returns:
        ad.AnnData
            The filtered AnnData object with updated `.obs` and `.obsm`.
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
    Convert an `.obs` column to categorical dtype.

    Args:
        adata : ad.AnnData
            The AnnData object.
        obs_key : str
            Column in `.obs` to be converted to categorical.
        drop_unused : bool, optional
            Whether to remove unused categories (default is True).
    """
    import pandas as pd

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
    Save the `.obsm` attribute of an AnnData object to an HDF5 file.

    Args:
        adata : anndata.AnnData
            The AnnData object containing the `.obsm` attribute to be saved.
        filename : str
            The path to the HDF5 file where `.obsm` data will be stored.

    Returns:
        None
            Saves `.obsm` data to the specified HDF5 file.
    """
    import h5py
    with h5py.File(filename, 'w') as f:
        obsm_group = f.create_group('obsm')
        for key, matrix in adata.obsm.items():
            obsm_group.create_dataset(key, data=matrix)


def load_obsm_from_hdf5(filename):
    """
    Load the `.obsm` attribute from an HDF5 file.

    Args:
        filename : str
            Path to the HDF5 file containing `.obsm` data.

    Returns:
        dict
            A dictionary where keys are `.obsm` names and values are corresponding matrices.
    """
    import h5py
    obsm = {}
    with h5py.File(filename, 'r') as f:
        obsm_group = f['obsm']
        for key in obsm_group.keys():
            obsm[key] = obsm_group[key][:]
    return obsm


def subset_adata_to_obsm_indices(adata, obsm):
    """
    Subset an AnnData object to match the indices present in `.obsm`.

    Args:
        adata : anndata.AnnData
            The original AnnData object.
        obsm : dict
            A dictionary containing `.obsm` data, where keys are embedding names, and values are arrays.

    Returns:
        anndata.AnnData
            A subsetted AnnData object that contains only the indices available in `.obsm`.
    """
    import numpy as np
    # Find the common indices across all obsm arrays
    indices = np.arange(adata.n_obs)
    for key in obsm:
        if obsm[key].shape[0] < indices.shape[0]:
            indices = indices[:obsm[key].shape[0]]

    # Subset the AnnData object
    adata_subset = adata[indices, :].copy()
    adata_subset.obsm = obsm
    return adata_subset



def get_adata_basis(adata, basis='X_pca'):
    """
    Retrieve a specific embedding from an AnnData object.

    Args:
        adata : ad.AnnData
            The AnnData object containing embeddings.
        basis : str, optional
            Key in `.obsm` specifying the embedding (default: "X_pca").

    Returns:
        np.ndarray
            The extracted embedding matrix.
    """
    import numpy as np
    if basis in adata.obsm:
        emb = adata.obsm[basis].astype(np.float32)
    elif basis == 'X':
        emb = adata.X.astype(np.float32)
    elif basis in adata.layers:
        emb = adata.layers[basis].astype(np.float32)
    else:
        raise ValueError(f"Embedding '{basis}' not found in adata.obsm or adata.layers.")
    
    return emb




def compute_meta_attributes(adata, groupby_key, attribute_key, method='majority_vote', meta_label_name=None):
    """
    Compute meta attributes for clusters (e.g., majority vote or mean value).

    Args:
        adata : ad.AnnData
            The AnnData object containing cell annotations.
        groupby_key : str
            The `.obs` column used to group cells (e.g., "leiden").
        attribute_key : str
            The `.obs` column to aggregate (e.g., "cell_type").
        method : str, optional
            Aggregation method: "majority_vote" for categorical, "average" for numerical (default: "majority_vote").
        meta_label_name : str, optional
            Name of the new meta attribute column (default: "meta_{attribute_key}").

    Returns:
        str
            Name of the newly added column in `.obs`.
    """
    import pandas as pd
    if meta_label_name is None:
        meta_label_name = f"meta_{attribute_key}"
    
    # Step 1: Compute meta attributes
    df = pd.DataFrame({attribute_key: adata.obs[attribute_key], groupby_key: adata.obs[groupby_key]})
    
    if method == 'majority_vote':
        # Categorical: compute most frequent value per group
        meta_attribute = df.groupby(groupby_key)[attribute_key].agg(lambda x: x.value_counts().idxmax())
    elif method == 'average':
        # Numeric: compute average per group
        meta_attribute = df.groupby(groupby_key)[attribute_key].mean()
    else:
        raise ValueError("Invalid method. Choose 'majority_vote' or 'average'.")
    
    # Map the meta attributes back to adata.obs
    adata.obs[meta_label_name] = adata.obs[groupby_key].map(meta_attribute)

    print(f"Added '{meta_label_name}' to adata.obs")
    return meta_label_name



def ordered_concat(
    adatas: List[ad.AnnData], 
    join: str = 'outer', 
    label: Optional[str] = None, 
    keys: Optional[List[str]] = None, 
    index_unique: Optional[str] = None,
    **kwargs
) -> ad.AnnData:
    """
    A wrapper for anndata.concat that preserves gene order based on the input list sequence.

    Instead of sorting the final variables (genes) alphabetically like the default
    anndata.concat, this function orders them sequentially: all genes from the
    first AnnData object appear first, followed by any *new* genes from the
    second object, and so on.

    Args:
        adatas: A list of AnnData objects to concatenate.
        join: Type of join ('inner' or 'outer'). Defaults to 'outer'.
        label: Column name to add to `.obs` identifying the origin of each cell.
        keys: Names for the `label` column. Inferred from `adatas.keys()` if it's a dict.
        index_unique: How to make indices unique. See anndata.concat documentation.
        **kwargs: Additional keyword arguments passed to anndata.concat.

    Returns:
        The concatenated AnnData object with a preserved, sequential gene order.
    """

    if not isinstance(adatas, list) or not adatas:
        raise ValueError("`adatas` must be a non-empty list of AnnData objects.")

    # --- Step 1: Determine the desired sequential gene order ---
    final_gene_order = []
    seen_genes = set()

    for adata in adatas:
        # Find genes in the current AnnData that we haven't seen yet
        new_genes = [gene for gene in adata.var_names if gene not in seen_genes]
        
        # Add these new genes to our final ordered list
        final_gene_order.extend(new_genes)
        
        # Update the set of seen genes
        seen_genes.update(adata.var_names)

    # --- Step 2: Perform the standard concatenation using the provided arguments ---
    # This creates a merged object, but with genes sorted alphabetically.
    # We pass all original arguments, including the list of adatas.
    concatenated_adata = ad.concat(
        adatas,
        join=join,
        label=label,
        keys=keys,
        index_unique=index_unique,
        **kwargs
    )

    # --- Step 3: Clean up any NaN values from the 'outer' join ---
    # This is important for data integrity and visualization.
    if hasattr(concatenated_adata.X, 'toarray'):
        concatenated_adata.X = concatenated_adata.X.toarray()
    concatenated_adata.X = np.nan_to_num(concatenated_adata.X, nan=0.0)
    
    for layer_key in concatenated_adata.layers.keys():
        if hasattr(concatenated_adata.layers[layer_key], 'toarray'):
            concatenated_adata.layers[layer_key] = concatenated_adata.layers[layer_key].toarray()
        concatenated_adata.layers[layer_key] = np.nan_to_num(concatenated_adata.layers[layer_key], nan=0.0)

    # --- Step 4: Re-index the AnnData object to enforce the correct order ---
    # This is the key step that fixes the sorting issue.
    # We use .copy() to ensure the final object is not a view.
    reordered_adata = concatenated_adata[:, final_gene_order].copy()

    return reordered_adata


def check_adata_X(adata, n_samples=100):
    """
    Quickly and memory-efficiently checks adata.X to guess if it contains
    raw counts or normalized values. This version is safe for both 
    in-memory and backed AnnData objects.
    """
    from scipy.sparse import issparse
    import numpy as np

    if adata.n_obs == 0:
        return 'empty'
    
    n_rows_to_sample = min(n_samples, adata.n_obs)
    X_sample = adata.X[:n_rows_to_sample, :]

    if issparse(X_sample):
        # We can now safely access .data on the small in-memory sample
        n_stored = X_sample.nnz
        if n_stored == 0:
            return 'empty'
        
        n_to_sample = min(n_samples, n_stored)
        data_sample = X_sample.data[:n_to_sample] # Simple slice is fine here
    else: # Dense case 
        # The subset is already small, so we can work with it directly
        non_zero_subset = X_sample[X_sample > 0]
        if non_zero_subset.size == 0:
            return 'empty'
        
        n_to_sample = min(n_samples, non_zero_subset.size)
        # Use simple slicing for consistency and speed
        data_sample = non_zero_subset[:n_to_sample]

    # Your simplified heuristic
    is_integer = np.all(np.isclose(data_sample, np.round(data_sample)))

    if is_integer:
        return 'raw'
    else:
        return 'normalized'
    


def filter_cells_min_genes(
    adata: sc.AnnData,
    min_genes: int = 10,
    verbose: bool = True,
) -> sc.AnnData:
    """
    Remove cells that express fewer than `min_genes` genes (non-zero counts)
    in the *current* adata.X matrix.

    Parameters
    ----------
    adata
        AnnData object (already subset to your HVGs, etc.).
    min_genes
        Minimum number of expressed genes a cell must have to be retained.
    verbose
        Whether to print a summary.

    Returns
    -------
    AnnData
        New AnnData containing only cells passing the filter.
    """
    
    from scipy.sparse import issparse
    # --- per-cell expressed-gene counts -------------------------------------
    if issparse(adata.X):
        n_expressed = np.asarray((adata.X > 0).sum(axis=1)).ravel()  # shape (n_cells,)
    else:
        n_expressed = (adata.X > 0).sum(axis=1)                      # ndarray (n_cells,)

    keep_mask = n_expressed >= min_genes
    n_before  = adata.n_obs
    n_after   = keep_mask.sum()

    if verbose:
        dropped = n_before - n_after
        print(f"ℹ️  Keeping cells with ≥{min_genes} expressed genes "
              f"({n_after}/{n_before} kept, {dropped} dropped).")

    # --- return filtered AnnData --------------------------------------------
    return adata[keep_mask].copy()




def load_anndata_from_dir(sample_dir):
    """
    Load Seurat-exported directory into an AnnData object (CSR format).

    Expected structure:
      sample_dir/
        counts.mtx or data.mtx   # main matrix
        obs.tsv                  # cell metadata
        var.tsv                  # gene metadata
        [other .mtx files]       # extra layers (e.g., counts.mtx, data.mtx)
        [*_embeddings.tsv]       # embeddings (optional)

    Returns
    -------
    adata : anndata.AnnData
        Loaded AnnData object with CSR matrices in X and layers.
    """
    import os
    import pandas as pd
    import anndata as ad
    from scipy.io import mmread
    from scipy.sparse import csr_matrix
    # --- 1. Identify matrix files
    all_files = os.listdir(sample_dir)
    layer_files = [f for f in all_files if f.endswith('.mtx')]

    # --- 2. Choose main matrix
    main_mtx_file = 'data.mtx' if 'data.mtx' in layer_files else 'counts.mtx'
    if main_mtx_file not in layer_files:
        raise ValueError(f"No main matrix file found in {sample_dir}. Expected 'data.mtx' or 'counts.mtx'.")

    # --- 3. Load matrices (convert to CSR)
    main_mtx = csr_matrix(mmread(os.path.join(sample_dir, main_mtx_file)).T)  # cells × genes

    # --- 4. Load obs / var metadata
    obs = pd.read_csv(os.path.join(sample_dir, 'obs.tsv'), sep='\t', index_col=0)
    var = pd.read_csv(os.path.join(sample_dir, 'var.tsv'), sep='\t', index_col=0)

    # --- 5. Build AnnData
    adata = ad.AnnData(X=main_mtx, obs=obs, var=var)

    # --- 6. Additional layers (convert each to CSR)
    for lf in layer_files:
        if lf != main_mtx_file:
            layer_name = lf.replace('.mtx', '')
            layer_mtx = csr_matrix(mmread(os.path.join(sample_dir, lf)).T)
            adata.layers[layer_name] = layer_mtx

    # --- 7. Load dimensional reductions (optional)
    emb_files = [f for f in all_files if f.endswith('_embeddings.tsv')]
    for ef in emb_files:
        red_name = ef.replace('_embeddings.tsv', '')
        emb_df = pd.read_csv(os.path.join(sample_dir, ef), sep='\t', index_col=0)
        adata.obsm[f'X_{red_name}'] = emb_df.values

    return adata


def _to_csr(x):
    from scipy.sparse import csr_matrix, issparse

    if issparse(x):
        return x.tocsr()
    # dense → csr
    return csr_matrix(x)

def _mmwrite_genes_by_cells(path, M):
    """
    Write a matrix as genes x cells Matrix Market (.mtx).
    AnnData stores X as cells x genes; we transpose on disk.
    """
    from scipy.io import mmwrite
    M = _to_csr(M)
    GxC = M.T  # genes x cells on disk
    with open(path, "wb") as f:
        mmwrite(f, GxC, field="real")  # "real" works for counts/normalized

def export_anndata_to_dir(adata: ad.AnnData, out_dir: str):
    """
    Export AnnData to a simple directory:
      - obs.tsv, var.tsv
      - data.mtx if adata.layers['data'] exists; otherwise uses adata.X as data.mtx
      - counts.mtx if adata.layers['counts'] exists
      - ALL obsm entries -> <name>_embeddings.tsv (name without leading 'X_')
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    # --- obs / var ---
    adata.obs.to_csv(p / "obs.tsv", sep="\t")
    adata.var.to_csv(p / "var.tsv", sep="\t")

    # --- main matrices ---
    # Prefer true normalized 'data' for data.mtx; else write X as data.mtx
    if "data" in adata.layers:
        _mmwrite_genes_by_cells(p / "data.mtx", adata.layers["data"])
    else:
        _mmwrite_genes_by_cells(p / "data.mtx", adata.X)

    # If counts exists, write it as well (in addition)
    if "counts" in adata.layers:
        _mmwrite_genes_by_cells(p / "counts.mtx", adata.layers["counts"])

    # --- write ALL obsm to <name>_embeddings.tsv
    for key, val in adata.obsm.items():
        # Convert to DataFrame with cells as index
        if isinstance(val, np.ndarray):
            cols = [f"{key}_{i+1}" for i in range(val.shape[1])]
            df = pd.DataFrame(val, index=adata.obs_names, columns=cols)
        else:
            df = pd.DataFrame(val)
            # ensure correct index
            if df.shape[0] == adata.n_obs:
                df.index = adata.obs_names
        
        df.to_csv(p / f"{key}_embeddings.tsv", sep="\t")
