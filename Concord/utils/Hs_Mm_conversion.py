from gseapy import Biomart
import pandas as pd
from .. import logger

# Helper function to chunk lists
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def get_mouse_genes(human_genes, return_type=None, chunk_size=500):
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
