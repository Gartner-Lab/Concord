import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scanpy as sc
from .. import logger

class QuotedString(str):
    pass

    def quoted_scalar_dumper(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

def convert_to_sparse_r_matrix(matrix):
    """
    Converts a SciPy sparse matrix to an R `dgCMatrix` object.

    Args:
        matrix (scipy.sparse matrix): A SciPy sparse matrix in COO, CSR, or CSC format.

    Returns:
        rpy2.robjects.RObject: An R sparse `dgCMatrix` object.

    Raises:
        ValueError: If the input matrix is not sparse.
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    # Import the required R package for sparse matrices
    Matrix = importr('Matrix')
    if sp.issparse(matrix):
        matrix_coo = matrix.tocoo()  # Convert to COO format if not already
        sparse_matrix_r = Matrix.sparseMatrix(
            i=ro.IntVector(matrix_coo.row + 1),  # R is 1-indexed
            j=ro.IntVector(matrix_coo.col + 1),
            x=ro.FloatVector(matrix_coo.data),
            dims=ro.IntVector(matrix_coo.shape)
        )
    else:
        raise ValueError("The input matrix is not sparse. Please provide a sparse matrix.")

    return sparse_matrix_r


def get_projection_df(adata, key):
    """
    Safely creates a DataFrame from an anndata.obsm entry,
    handling both pandas DataFrame and NumPy array inputs.
    """
    # Get the data from the .obsm slot
    data_source = adata.obsm[key]
    
    # np.asarray() efficiently gets the raw data from either input type
    raw_data = np.asarray(data_source)
    
    # Create the new DataFrame using the raw data
    proj_df = pd.DataFrame(
        raw_data,
        index=adata.obs.index,
        columns=[f"{key}_{i+1}" for i in range(raw_data.shape[1])]
    )
    return proj_df

def anndata_to_viscello(adata, output_dir, project_name="MyProject", organism='hsa', clist_only = False):
    """
    Converts an AnnData object to a VisCello project directory.

    Args:
        adata (AnnData): AnnData object containing single-cell data.
        output_dir (str): Directory where the VisCello project will be created.
        project_name (str, optional): Name of the project. Defaults to "MyProject".
        organism (str, optional): Organism code (e.g., 'hsa' for human). Defaults to 'hsa'.
        clist_only (bool, optional): Whether to generate only the clist file. Defaults to False.

    Returns:
        None

    Side Effects:
        - Creates a directory with the necessary files for VisCello.
        - Saves `eset.rds` (ExpressionSet), `config.yml`, and `clist.rds`.
    """
    # Import the required R packages
    import rpy2.robjects as ro
    from rpy2.robjects import ListVector  # Removed pandas2ri import (no longer needed globally)
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri  # Keep for converter

    base = importr('base')
    methods = importr('methods')
    biobase = importr('Biobase')
    # Create the output directory structure
    os.makedirs(output_dir, exist_ok=True)

    # Define the Cello class in R from Python
    ro.r('''
        setClass("Cello",
                slots = c(
                    name = "character",   # The name of the cello object
                    idx = "numeric",      # The index of the global cds object
                    proj = "list",        # The projections as a list of data frames
                    pmeta = "data.frame", # The local meta data
                    notes = "character"   # Other information to display to the user
                )
        )
    ''')
    
    if not clist_only:
        # Convert the expression matrix to a sparse matrix in R (no pandas conversion needed here)
        if 'counts' in adata.layers:
            exprs_sparse_r = convert_to_sparse_r_matrix(adata.layers['counts'].T)
        else:
            logger.info("Counts data (adata.layers['counts']) not found. Please make sure you save a copy of raw data in adata.layers['counts'].")
            exprs_sparse_r = convert_to_sparse_r_matrix(adata.X.T)
        
        # Convert the normalized expression matrix to a sparse matrix in R
        norm_exprs_sparse_r = convert_to_sparse_r_matrix(adata.X.T)
        
        # NEW: Wrap pandas-to-R conversions in context manager
        #fmeta = pd.DataFrame({'gene_short_name': adata.var.index}, index=adata.var.index)
        fmeta = adata.var.assign(gene_short_name=adata.var.index)
        with (ro.default_converter + pandas2ri.converter).context():
            annotated_pmeta = methods.new("AnnotatedDataFrame", data=ro.conversion.py2rpy(adata.obs))
            annotated_fmeta = methods.new("AnnotatedDataFrame", data=ro.conversion.py2rpy(fmeta))

        # Create the ExpressionSet object in R
        eset = methods.new(
            "ExpressionSet",
            assayData=ro.r['assayDataNew'](
                "environment", 
                exprs=exprs_sparse_r, 
                norm_exprs=norm_exprs_sparse_r
            ),
            phenoData=annotated_pmeta,
            featureData=annotated_fmeta
        )
        
        # Save the ExpressionSet as an RDS file
        rds_file = os.path.join(output_dir, "eset.rds")
        ro.r['saveRDS'](eset, file=rds_file)

        # Prepare and save the config.yml file
        config_content = f"""
            default:
                study_name: "{project_name}"
                study_description: ""
                organism: "{organism}"
                feature_name_column: "{fmeta.columns[0]}"
                feature_id_column: "{fmeta.columns[0]}"
            """
        
        config_file = os.path.join(output_dir, "config.yml")
        with open(config_file, 'w') as file:
            file.write(config_content.strip())
    
    # Prepare and save the clist object
    proj_list = {}
    # NEW: Wrap projection conversions in context manager (multiple py2rpy calls)
    with (ro.default_converter + pandas2ri.converter).context():
        for key in adata.obsm_keys():
            proj_df = get_projection_df(adata, key)
            proj_r_df = ro.conversion.py2rpy(proj_df)
            # change column name to valid R column name with make.names
            proj_r_df.colnames = base.make_names(proj_r_df.colnames)
            proj_list[key] = proj_r_df

    # Assign the proj list to the cello object
    proj_list_r = ListVector(proj_list)

    cell_index = ro.IntVector(range(1, adata.n_obs + 1))  # assuming all cells are used
    # Note: Cello creation doesn't need further pandas conversion here
    cello = methods.new("Cello", name="All cells", idx=cell_index, proj=proj_list_r)

    # Create the clist and save it
    clist = ListVector({"All cells": cello})

    clist_file = os.path.join(output_dir, "clist.rds")
    ro.r['saveRDS'](clist, file=clist_file)

    print(f"VisCello project created at {output_dir}")


def update_clist_with_subsets(global_adata, adata_subsets, viscello_dir, cluster_key = None):
    """
    Updates an existing VisCello clist with new subsets.

    Args:
        global_adata (AnnData): The full AnnData object.
        adata_subsets (dict): Dictionary mapping subset names to AnnData objects.
        viscello_dir (str): Path to the existing VisCello directory.
        cluster_key (str, optional): Key in `adata.obs` for cluster assignments. Defaults to None.

    Returns:
        None

    Side Effects:
        - Reads the existing `clist.rds` file from `viscello_dir`.
        - Adds new subsets as `Cello` objects to the clist.
        - Saves the updated `clist.rds` file in `viscello_dir`.
    """
    import os
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import ListVector, pandas2ri  # Added pandas2ri for converter
    from rpy2.robjects.packages import importr
    # Removed: pandas2ri.activate()

    # Define the Cello class in R from Python
    ro.r('''
        setClass("Cello",
                slots = c(
                    name = "character",   # The name of the cello object
                    idx = "numeric",      # The index of the global cds object
                    proj = "list",        # The projections as a list of data frames
                    pmeta = "data.frame", # The local meta data
                    notes = "character"   # Other information to display to the user
                )
        )
    ''')

    # Load R packages
    base = importr('base')
    methods = importr('methods')

    # Load the existing clist
    clist_file = os.path.join(viscello_dir, "clist.rds")
    print(f"Loading existing clist from: {viscello_dir}")
    existing_clist = ro.r['readRDS'](str(clist_file))  # Convert pathlib.Path to str

    # Prepare updated clist
    clist_objects = dict(existing_clist.items())  # Convert R ListVector to a Python dictionary

    # Process each subset and add to clist
    for subset_name, adata_subset in adata_subsets.items():
        print(f"Processing subset: {subset_name}")

        # Get the global indices of the subset cells
        global_indices = global_adata.obs.index.get_indexer(adata_subset.obs.index) + 1  # R uses 1-based indexing

        # Prepare proj slot (latent spaces)
        proj_list = {}
        # NEW: Wrap projection conversions in context manager
        with (ro.default_converter + pandas2ri.converter).context():
            for key in adata_subset.obsm.keys():
                proj_df = get_projection_df(adata_subset, key)
                proj_r_df = ro.conversion.py2rpy(proj_df)
                proj_r_df.colnames = base.make_names(proj_r_df.colnames)  # Ensure valid R column names
                proj_list[key] = proj_r_df

        # Convert proj_list to an R ListVector
        proj_list_r = ListVector(proj_list)

        # Prepare pmeta slot (clustering results)
        if cluster_key:
            clustering_results = adata_subset.obs[[cluster_key]]
            # NEW: Wrap pmeta conversion in context manager
            with (ro.default_converter + pandas2ri.converter).context():
                pmeta_r = ro.conversion.py2rpy(clustering_results)
        else:
            # Make empty data frame with same rownames as adata_subset.obs
            pmeta_r = ro.r['data.frame'](rownames=ro.StrVector(adata_subset.obs.index))
            
        # Create Cello object for the subset
        cello = methods.new(
            "Cello",
            name=subset_name,
            idx=ro.IntVector(global_indices),  # Global indices
            proj=proj_list_r,  # Projections
            pmeta=pmeta_r,  # Clustering metadata
            notes=f"Subset: {subset_name}"
        )

        # Add the new Cello object to the clist
        clist_objects[subset_name] = cello

    # Convert updated clist_objects to an R ListVector
    updated_clist = ListVector(clist_objects)

    # Save the updated clist
    output_file = os.path.join(viscello_dir, "clist.rds")
    ro.r['saveRDS'](updated_clist, file=str(output_file))  # Convert pathlib.Path to str
    print(f"Updated clist saved to: {output_file}")

