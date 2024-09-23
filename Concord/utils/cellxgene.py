
def list_datasets_by_criteria(cell_type=None, tissue=None, developmental_stage=None, group_by_columns=None):
    import cellxgene_census
    import anndata
    import os
    import tiledbsoma as soma
    import cellxgene_census.experimental.ml as census_ml

    # Initialize census
    census = cellxgene_census.open_soma()

    try:
        # Read datasets metadata
        census_datasets = (
            census["census_info"]["datasets"]
            .read()
            .concat()
            .to_pandas()
        )

        # Build the value filter for the given criteria
        value_filters = []
        if cell_type:
            value_filters.append(f"cell_type == '{cell_type}'")
        if tissue:
            value_filters.append(f"tissue == '{tissue}'")
        if developmental_stage:
            value_filters.append(f"developmental_stage == '{developmental_stage}'")
        value_filter = " and ".join(value_filters)

        # Read observation data filtered by the given criteria
        obs_df = (
            census["census_data"]["homo_sapiens"]
            .obs.read(
                value_filter=value_filter + " and is_primary_data == True"
            )
            .concat()
            .to_pandas()
        )

        # Define default group_by_columns if not provided
        if group_by_columns is None:
            group_by_columns = ["dataset_id"]

        # Group and tabulate data by the specified columns
        dataset_cell_counts = obs_df.groupby(group_by_columns).size().reset_index(name='cell_counts')

        # Merge with the datasets metadata
        dataset_cell_counts = dataset_cell_counts.merge(census_datasets, on="dataset_id")

        return dataset_cell_counts

    finally:
        # Ensure census is closed
        census.close()


import cellxgene_census
import anndata


def download_anndata(organism="Homo sapiens", dataset_id=None, cell_type=None, tissue=None, disease=None, developmental_stage=None,
                     output_path='output.h5ad'):
    # Initialize census
    census = cellxgene_census.open_soma()

    try:
        # Build the value filter for the given criteria
        value_filters = []
        if dataset_id:
            value_filters.append(f"dataset_id == '{dataset_id}'")
        if cell_type:
            value_filters.append(f"cell_type == '{cell_type}'")
        if tissue:
            value_filters.append(f"tissue == '{tissue}'")
        if disease:
            value_filters.append(f"disease == '{disease}'")
        if developmental_stage:
            value_filters.append(f"developmental_stage == '{developmental_stage}'")
        value_filter = " and ".join(value_filters)

        # Download the AnnData object
        adata = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
            measurement_name="RNA",
            X_name="raw",
            obs_value_filter=value_filter + " and is_primary_data == True"
        )
        print(f"Loaded data shape: {adata.shape}")
        # Save the AnnData object to disk
        adata.write_h5ad(output_path)
        print(f"AnnData object saved to {output_path}")

    finally:
        # Ensure census is closed
        census.close()



