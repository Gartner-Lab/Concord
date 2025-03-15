# CONCORD: COntrastive learNing for Cross-dOmain Reconciliation and Discovery

Qin Zhu, Gartner Lab, UCSF

## Description

Resolving the intricate structure of the cellular state landscape from single-cell RNA sequencing (scRNAseq) experiments remains an outstanding challenge, compounded by technical noise and systematic discrepancies—often referred to as batch effects—across experimental systems and replicate. To address this, we introduce **CONCORD (COntrastive learNing for Cross-dOmain Reconciliation and Discovery)**, a self-supervised contrastive learning framework designed for robust **dimensionality reduction** and **data integration** in single-cell analysis. The core innovation of CONCORD lies in its probabilistic, dataset- and neighborhood-aware sampling strategy, which enhances contrastive learning by simultaneously improving the resolution of cell states and mitigating batch artifacts. Operated in a fully unsupervised manner, CONCORD generates **denoised cell encodings** that faithfully preserve key biological structures, from fine-grained distinctions among closely related cell states to large-scale topological organizations. The resulting high-resolution cell atlas seamlessly integrates data across experimental batches, technologies, and species. Additionally, CONCORD’s latent space capture biologically meaningful **gene programs**, enabling the exploration of regulatory mechanisms underlying cell state transitions and subpopulation heterogeneity. We demonstrate the utility of CONCORD on a range of topological structures and biological contexts, underscoring its potential to extract meaningful insights from both existing and future single-cell datasets.

**Full Documentation available at https://qinzhu.github.io/Concord_documentation/.**

---

## Installation

### 1. Clone the Concord repository and set up environment:

```bash
git clone git@github.com:Gartner-Lab/Concord.git
```

It is recommended to use conda (https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to create and set up a virtual environment for Concord.

### 2. Install PyTorch:

You must install the correct version of PyTorch based on your system's CUDA setup. Please follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/) to install the appropriate version of PyTorch for CUDA or CPU.

Example (for CPU version):
```bash
pip install torch torchvision torchaudio
```

### 3. Install dependencies:

Navigate to the Concord directory and install the required dependencies:

```bash
cd path_to_Concord
pip install -r requirements.txt
```

### 4. Install Concord:
Build and install Concord:

```bash
python -m build
pip install dist/Concord-0.9.0-py3-none-any.whl
```

### 5. (Optional) Install FAISS for accelerated KNN search (not recommended for Mac):

Install FAISS for fast nearest-neighbor searches for large datasets. Note if you are using Mac, you should turn faiss off by specifying `cur_ccd = ccd.Concord(adata=adata, input_feature=feature_list, use_faiss=False, device=device)` when running Concord, unless you are certain faiss runs with no problem.

- **FAISS with GPU**:
  ```bash
  pip install faiss_gpu
  ```
- **FAISS with CPU**:
  ```bash
  pip install faiss_cpu
  ```

### 6. (Optional) Install optional dependencies:

Concord offers additional functionality through optional dependencies. You can install them via:
```bash
pip install -r requirements_optional.txt
```

### 7. (Optional) Integration with VisCello:

Concord integrates with **VisCello**, a tool for interactive visualization. To explore results interactively, visit [VisCello GitHub](https://github.com/kimpenn/VisCello) and refer to the full documentation for more information.

You will also need the rpy2 package installed via:
```bash
pip install rpy2
```

---

## Getting Started

Concord integrates seamlessly with `anndata` objects. 
Single-cell datasets, such as 10x Genomics outputs, can easily be loaded into an `AnnData` object using the [`Scanpy`](https://scanpy.readthedocs.io/) package. If you're using R and have data in a `Seurat` object, you can convert it to `anndata` format by following this [tutorial](https://qinzhu.github.io/Concord_documentation/). 
In this quick-start example, we'll demonstrate CONCORD using the `pbmc3k` dataset provided by the `scanpy` package.

### Load package and data

```python
# Load required packages
import Concord as ccd
import scanpy as sc
import torch
# Load and prepare example data
adata = sc.datasets.pbmc3k_processed()
adata = adata.raw.to_adata()  # Store raw counts in adata.X, by default Concord will run standard total count normalization and log transformation internally, not necessary if you want to use your normalized data in adata.X, if so, specify 'X' in cur_ccd.encode_adata(input_layer_key='X', output_key='Concord')
```

### Run CONCORD:

```python
# Set device to cpu or to gpu (if your torch has been set up correctly to use GPU), for mac you can use either torch.device('mps') or torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# (Optional) Select top variably expressed/accessible features for analysis (other methods besides seurat_v3 available)
feature_list = ccd.ul.select_features(adata, n_top_features=5000, flavor='seurat_v3')

# Initialize Concord with an AnnData object, skip input_feature to use all features
cur_ccd = ccd.Concord(adata=adata, input_feature=feature_list, device=device) 

# If integrating data across batch, simply add the domain_key argument to indicate the batch key in adata.obs
# cur_ccd = ccd.Concord(adata=adata, input_feature=feature_list, domain_key='batch', device=device) 

# Encode data, saving the latent embedding in adata.obsm['Concord']
cur_ccd.encode_adata(output_key='Concord')
```

### Visualization:

CONCORD latent embeddings can be directly used for downstream analyses such as visualization with UMAP and t-SNE or constructing k-nearest neighbor (kNN) graphs. Unlike PCA, it is important to utilize the full CONCORD latent embedding in downstream analyses, as each dimension is designed to capture meaningful and complementary aspects of the underlying data structure.

```python
ccd.ul.run_umap(adata, source_key='Concord', result_key='Concord_UMAP', n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')

# Plot the UMAP embeddings
color_by = ['n_genes', 'louvain'] # Choose which variables you want to visualize
ccd.pl.plot_embedding(
    adata, basis='Concord_UMAP', color_by=color_by, figsize=(10, 5), dpi=600, ncols=2, font_size=6, point_size=10, legend_loc='on data',
    save_path='Concord_UMAP.png'
)
```

The latent space produced by CONCORD often capture complex biological structures that may not be fully visualized in 2D projections. We recommend exploring the latent space using a 3D UMAP to more effectively capture and examine the intricacies of the data. For example:

```python
ccd.ul.run_umap(adata, source_key='Concord', result_key='Concord_UMAP_3D', n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean')

# Plot the 3D UMAP embeddings
col = 'louvain'
fig = ccd.pl.plot_embedding_3d(
    adata, basis='Concord_UMAP_3D', color_by=col, 
    save_path='Concord_UMAP_3D.html',
    point_size=3, opacity=0.8, width=1500, height=1000
)
```

---

## Citation

Please cite the preprint here: [Insert citation link].

