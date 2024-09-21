
print("importing")
import scanpy as sc
import time
from pathlib import Path
import torch
import Concord as ccd
import warnings
warnings.filterwarnings('ignore')
print("imported")

proj_name = "concord_treg"
save_dir = f"/wynton/home/gartner/zhuqin/Concord/save/dev_{proj_name}-{time.strftime('%b%d')}/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
seed = 0
ccd.ul.set_seed(seed)

data_path = "/wynton/home/gartner/zhuqin/Concord/data/treg_data/R_treg_graph_ad.h5ad"
adata = sc.read(
    data_path
)
adata.obs['T_subtype_refined']=adata.obs['T_subtype_refined'].cat.remove_unused_categories()

feature_list = ccd.ul.select_features(adata, n_top_features=3000, flavor='seurat_v3')

params = {
    "project_name": proj_name,
    "input_feature": feature_list,
    "batch_size": 64,
    "n_epochs": 3,
    "lr": 1e-3,
    "latent_dim": 32,
    "encoder_dims": [256],
    "decoder_dims": [256],
    "augmentation_mask_prob": 0.5,
    "dropout_prob": 0.1,
    "use_decoder": True,
    "decoder_weight": 1.0,
    "use_classifier": False,
    "classifier_weight": 1.0,
    "clr_temperature":0.5,
    "domain_embedding_dim":8,
    "use_importance_mask": True,
    "importance_penalty_type": 'L1',
    "importance_penalty_weight": 0,
    "domain_key": 'dataset.ident',
    "class_key": None,
    "sampler_emb": "X_pca",
    "sampler_knn": 128,
    "p_intra_knn": 0.5,
    "p_intra_domain": None,
    "min_p_intra_domain": 0.9,
    "pca_n_comps": 50,
    "use_faiss": True,
    "use_ivf": True,
    "ivf_nprobe": 10,
    "inplace": False,
    "save_dir": save_dir,
    "device": device,
    "verbose": True
}
output_key = "Concord"
file_suffix = f"{time.strftime('%b%d-%H%M')}"

cur_ccd = ccd.Concord(adata=adata, **params)
cur_ccd.encode_adata(input_layer_key="X_log1p", output_key=output_key)
obsm_filename = save_dir / f"obsm_{file_suffix}.h5"
ccd.ul.save_obsm_to_hdf5(cur_ccd.adata, obsm_filename)
adata.obsm[output_key] = cur_ccd.adata.obsm[output_key] # If not inplace

ccd.ul.run_umap(adata, source_key=output_key, umap_key=f'{output_key}_UMAP', n_components=2, n_epochs=500, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=seed, use_cuml=False)
sc.pp.neighbors(adata, use_rep="Concord")
sc.tl.leiden(adata, resolution=2.0)
show_cols = ['dataset.ident', 'nFeature_RNA', 'T_subtype_refined', 'leiden']
show_emb = 'Concord_UMAP'

ccd.pl.plot_embedding(
    adata, show_emb, show_cols, figsize=(10,8), dpi=600, ncols=2, font_size=6, point_size=3, legend_loc='on data',
    save_path=save_dir / f"embeddings_{show_emb}_{file_suffix}.png"
)
