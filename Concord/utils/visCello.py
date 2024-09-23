

import os
import pandas as pd
import anndata
import yaml
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, ListVector
from rpy2.robjects.packages import importr
import scipy.sparse as sp
import scanpy as sc
from .. import logger

class QuotedString(str):
    pass

    def quoted_scalar_dumper(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

def convert_to_sparse_r_matrix(matrix):
    """
    Converts a sparse matrix in Python to a sparse dgCMatrix in R.

    Parameters:
    - matrix: A scipy sparse matrix (in COO, CSR, or CSC format) or a dense numpy array.

    Returns:
    - A sparse dgCMatrix R object.
    """
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



def anndata_to_viscello(adata, output_dir, project_name="MyProject", organism='hsa', clist_only = False):
    """
    Converts an AnnData object to a VisCello cello folder with the necessary files.

    Parameters:
    - adata: AnnData object containing your single-cell data.
    - output_dir: The directory where the VisCello folder will be created.
    - project_name: Name of the project (used for the folder name).
    - organism: Organism code (e.g., 'hsa' for human).
    """
    # Import the required R packages
    pandas2ri.activate()
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
        # Convert the expression matrix to a sparse matrix in R
        if 'counts' in adata.layers:
            exprs_sparse_r = convert_to_sparse_r_matrix(adata.layers['counts'].T)
        else:
            exprs_sparse_r = convert_to_sparse_r_matrix(adata.X.T)
        
        # Convert the normalized expression matrix (adata.layers['X_log1p']) to a sparse matrix in R
        if 'X_log1p' in adata.layers:
            norm_exprs_sparse_r = convert_to_sparse_r_matrix(adata.layers['X_log1p'].T)
        else:
            # TODO use preprocessor to check if data is already normalized and log transformed.
            logger.info("Normalized expression data (adata.layers['X_log1p']) not found. Renormalize and log transforming.")
            tmp_adata = adata.copy()
            if 'counts' in adata.layers:
                tmp_adata.X = tmp_adata.layers['counts']
            sc.pp.normalize_total(tmp_adata, target_sum=1e4)
            sc.pp.log1p(tmp_adata)
            norm_exprs_sparse_r = convert_to_sparse_r_matrix(tmp_adata.X.T)
        
        # Convert phenoData and featureData to R
        fmeta = pd.DataFrame({'gene_short_name': adata.var.index}, index=adata.var.index)
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
    for key in adata.obsm_keys():
        proj_df = pd.DataFrame(adata.obsm[key], index=adata.obs.index, columns=[f"{key}_{i+1}" for i in range(adata.obsm[key].shape[1])])
        proj_r_df = ro.conversion.py2rpy(proj_df)
        proj_list[key] = proj_r_df

    # Assign the proj list to the cello object
    proj_list_r = ListVector(proj_list)

    cell_index = ro.IntVector(range(1, adata.n_obs + 1))  # assuming all cells are used
    cello = methods.new("Cello", name="All cells", idx=cell_index, proj=proj_list_r)

    # Create the clist and save it
    clist = ListVector({"All cells": cello})

    clist_file = os.path.join(output_dir, "clist.rds")
    ro.r['saveRDS'](clist, file=clist_file)

    print(f"VisCello project created at {output_dir}")
