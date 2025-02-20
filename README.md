# Concord: Contrastive Learning for Cross-domain Reconciliation and Discovery

Qin Zhu, Gartner Lab, UCSF

## Description

CONCORD (COntrastive learNing for Cross-dOmain Reconciliation and Discovery) is a novel machine learning framework that leverages contrastive learning, masked autoencoders, and a unique batch construction strategy using data-aware sampling. CONCORD learns an encoding of cells that captures the cell state manifold, revealing both local and global structures. The resulting high-resolution atlas of cell states and trajectories is coherent across different domains, such as batches, technologies, and species. 

**Full Documentation available at [insert documentation link].**

---

## Installation

### 1. Clone the Concord repository and set up environment:

```bash
git clone git@github.com:Gartner-Lab/Concord.git
```

It is recommended to use conda (https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to create and set up virtual environment for Concord.

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

---

## Quick Start

Concord seamlessly works with `anndata` objects. Here’s an example run:

```python
import Concord as ccd
import scanpy as sc
import torch

adata = sc.datasets.pbmc3k_processed()
adata = adata.raw.to_adata()  # Store raw counts in adata.X, by default Concord will run standard total count normalization and log transformation internally

# Set device to cpu or to gpu (if your torch has been set up correctly to use GPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Select top variably expressed/accessible features for analysis (other methods besides seurat_v3 available)
feature_list = ccd.ul.select_features(adata, n_top_features=5000, flavor='seurat_v3')

# Initialize Concord with an AnnData object, skip input_feature default to all features
cur_ccd = ccd.Concord(adata=adata, input_feature=feature_list, device=device) 
# If integrating data across batch, simply add the domain_key argument
# cur_ccd = ccd.Concord(adata=adata, input_feature=feature_list, domain_key='batch', device=device) 

# Encode data, saving the latent embedding in adata.obsm['Concord']
cur_ccd.encode_adata(input_layer_key='X_log1p', output_key='Concord')
```

### Visualize Results:

We recommend using UMAP to visualize Concord embeddings:

```python
ccd.ul.run_umap(adata, source_key='Concord', umap_key='Concord_UMAP', n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')

# Plot the UMAP embeddings
color_by = ['n_genes', 'louvain'] # Choose which variables you want to visualize
ccd.pl.plot_embedding(
    adata, basis='Concord_UMAP', color_by=color_by, figsize=(10, 5), dpi=600, ncols=2, font_size=6, point_size=10, legend_loc='on data',
    save_path='Concord_UMAP.png'
)
```

### 3D Visualization:
For complex structures, 3D UMAP may provide better insights:

```python
ccd.ul.run_umap(adata, source_key='Concord', result_key='Concord_UMAP_3D', n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean')

# Plot the 3D UMAP embeddings
col = 'louvain'
ccd.pl.plot_embedding_3d(
    adata, basis='Concord_UMAP_3D', color_by=col,
    save_path='Concord_UMAP_3D.html',
    point_size=10, opacity=0.8, width=1500, height=1000
)
```

---

## Citation

Please cite the preprint here: [Insert citation link].

