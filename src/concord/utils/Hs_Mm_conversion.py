import pandas as pd
from .. import logger

# Helper function to chunk lists
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def get_mouse_genes(human_genes, return_type=None, chunk_size=500):
    try:
        from gseapy import Biomart
    except ImportError:
        raise ImportError("gseapy is required for this method. Please install it using 'pip install gseapy'.")

    bm = Biomart()
    results = []

    total_processed = 0

    for chunk in chunk_list(human_genes, chunk_size):
        h2m_chunk = bm.query(dataset='hsapiens_gene_ensembl',
                             filters={'external_gene_name': chunk},
                             attributes=['external_gene_name', 'mmusculus_homolog_associated_gene_name'])
        results.append(h2m_chunk)
        total_processed += len(chunk)
        logger.info(f"Processed {total_processed} human genes to mouse orthologs.")

    h2m = pd.concat(results, ignore_index=True)

    if return_type == 'dict':
        return dict(zip(h2m['external_gene_name'], h2m['mmusculus_homolog_associated_gene_name']))
    elif return_type == 'pandas':
        return h2m
    else:
        return h2m['mmusculus_homolog_associated_gene_name'].dropna().unique().tolist()

def get_human_genes(mouse_genes, return_type=None, chunk_size=100):
    try:
        from gseapy import Biomart
    except ImportError:
        raise ImportError("gseapy is required for this method. Please install it using 'pip install gseapy'.")
    bm = Biomart()
    results = []
    total_processed = 0

    for chunk in chunk_list(mouse_genes, chunk_size):
        m2h_chunk = bm.query(dataset='mmusculus_gene_ensembl',
                             filters={'external_gene_name': chunk},
                             attributes=['external_gene_name', 'hsapiens_homolog_associated_gene_name'])
        results.append(m2h_chunk)
        total_processed += len(chunk)
        logger.info(f"Processed {total_processed} mouse genes to human orthologs.")

    m2h = pd.concat(results, ignore_index=True)

    if return_type == 'dict':
        return dict(zip(m2h['external_gene_name'], m2h['hsapiens_homolog_associated_gene_name']))
    elif return_type == 'pandas':
        return m2h
    else:
        return m2h['hsapiens_homolog_associated_gene_name'].dropna().unique().tolist()


def get_mouse_genes_offline(human_genes, orthologs, return_type=None, keep_all=False):
    """
    Map human genes to mouse orthologs using the provided ortholog table.

    Parameters:
        human_genes (list): List of human gene symbols to map.
        orthologs (pd.DataFrame): The loaded ortholog table.
        return_type (str): Output format ('dict', 'pandas', or None for a list of unique mouse genes).
        keep_all (bool): If True, return all mapped mouse genes for each human gene.
                                If False, return only the first mapped mouse gene.

    Returns:
        dict, pd.DataFrame, or list: Mouse gene orthologs in the specified format.
    """
    # Filter rows for human and mouse genes
    df_human = orthologs[orthologs["Common Organism Name"] == "human"]
    df_mouse = orthologs[orthologs["Common Organism Name"] == "mouse, laboratory"]

    # Merge human and mouse gene data on 'DB Class Key'
    human_to_mouse = df_human.merge(
        df_mouse,
        on="DB Class Key",
        suffixes=("_human", "_mouse")
    )

    # Filter for human genes of interest
    mapping = human_to_mouse[human_to_mouse["Symbol_human"].isin(human_genes)][["Symbol_human", "Symbol_mouse"]]

    # Group and apply keep_all logic
    if keep_all:
        grouped = mapping.groupby("Symbol_human")["Symbol_mouse"].apply(list)
    else:
        grouped = mapping.groupby("Symbol_human")["Symbol_mouse"].first()

    # Return in the desired format
    if return_type == "dict":
        return grouped.to_dict()
    elif return_type == "pandas":
        return grouped.reset_index()
    else:
        return grouped.explode().unique().tolist()


def get_human_genes_offline(mouse_genes, orthologs, return_type=None, keep_all=False):
    """
    Map mouse genes to human orthologs using the provided ortholog table.

    Parameters:
        mouse_genes (list): List of mouse gene symbols to map.
        orthologs (pd.DataFrame): The loaded ortholog table.
        return_type (str): Output format ('dict', 'pandas', or None for a list of unique human genes).
        keep_all (bool): If True, return all mapped human genes for each mouse gene.
                                If False, return only the first mapped human gene.

    Returns:
        dict, pd.DataFrame, or list: Human gene orthologs in the specified format.
    """
    # Filter rows for human and mouse genes
    df_human = orthologs[orthologs["Common Organism Name"] == "human"]
    df_mouse = orthologs[orthologs["Common Organism Name"] == "mouse, laboratory"]

    # Merge human and mouse gene data on 'DB Class Key'
    mouse_to_human = df_mouse.merge(
        df_human,
        on="DB Class Key",
        suffixes=("_mouse", "_human")
    )

    # Filter for mouse genes of interest
    mapping = mouse_to_human[mouse_to_human["Symbol_mouse"].isin(mouse_genes)][["Symbol_mouse", "Symbol_human"]]

    # Group and apply keep_all logic
    if keep_all:
        grouped = mapping.groupby("Symbol_mouse")["Symbol_human"].apply(list)
    else:
        grouped = mapping.groupby("Symbol_mouse")["Symbol_human"].first()

    # Return in the desired format
    if return_type == "dict":
        return grouped.to_dict()
    elif return_type == "pandas":
        return grouped.reset_index()
    else:
        return grouped.explode().unique().tolist()
