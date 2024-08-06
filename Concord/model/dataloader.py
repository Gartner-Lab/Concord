import torch
from .classSampler import ClassSampler
from .neighborhoodSampler import NeighborhoodSampler
from .anndataset import AnnDataset
from torch.utils.data import DataLoader, random_split, Subset
from .. import logger


def create_dataloader(dataset, batch_size, sampler_mode, device, drop_last=True, emb_key='encoded',
                      sampler_knn=300, p_intra_knn=0.3, p_intra_domain=1.0,
                      class_weights=None, p_intra_class=0.3,
                      use_faiss=True, use_ivf=False, ivf_nprobe=8, ):

    if sampler_mode == "domain":
        sampler = ClassSampler(dataset, batch_size=batch_size, class_weights=None, p_intra=None, drop_last=drop_last, device=device)
        dataloader = DataLoader(dataset, batch_sampler=sampler)
    elif sampler_mode == "class":
        sampler = ClassSampler(dataset, batch_size=batch_size, class_weights=class_weights,
                               p_intra=p_intra_class, drop_last=drop_last, device=device)
        dataloader = DataLoader(dataset, batch_sampler=sampler)
    elif sampler_mode == "neighborhood":
        if p_intra_knn is None:
            raise ValueError("p_intra_knn is required for neighborhood sampler.")
        sampler = NeighborhoodSampler(dataset, batch_size=batch_size, emb_key=emb_key,
                                      sampler_knn=sampler_knn, p_intra_knn=p_intra_knn, p_intra_domain=p_intra_domain,
                                      use_faiss=use_faiss, use_ivf=use_ivf, ivf_nprobe=ivf_nprobe, device=device)
        dataloader = DataLoader(dataset, batch_sampler=sampler)
    elif sampler_mode is None:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    else:
        raise ValueError("Unknown sampler, please choose sampler mode from 'domain', 'class', 'neighborhood', or None.")
    logger.info(f"Created DataLoader with {len(dataloader)} batches.")
    return dataloader

def anndata_to_dataloader(adata, input_layer_key, domain_key,
                          class_key=None,
                          extra_keys=None,
                          train_frac=0.9,
                          train_indices=None, val_indices=None,
                          batch_size=32,
                          sampler_mode="domain",
                          emb_key=None,
                          sampler_knn=300,
                          p_intra_knn=0.3,
                          p_intra_domain=1.0,
                          use_faiss=True, use_ivf=False, ivf_nprobe=8,
                          class_weights=None,
                          p_intra_class = 0.3,
                          drop_last=True,
                          preprocess=None,
                          device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if preprocess:
        preprocess(adata)

    dataset = AnnDataset(adata, input_layer_key=input_layer_key, domain_key=domain_key,
                             class_key=class_key, extra_keys=extra_keys, device=device)
    data_structure = dataset.get_data_structure()

    dataloader_kwargs = {
        'batch_size': batch_size,
        'sampler_mode': sampler_mode,
        'emb_key': emb_key,
        'sampler_knn': sampler_knn,
        'p_intra_knn': p_intra_knn,
        'p_intra_domain': p_intra_domain,
        'use_faiss': use_faiss,
        'use_ivf': use_ivf,
        'ivf_nprobe': ivf_nprobe,
        'class_weights': class_weights,
        'p_intra_class': p_intra_class,
        'drop_last': drop_last,
        'device': device
    }

    if train_frac == 1.0:
        # Create a single DataLoader without splitting
        full_dataloader = create_dataloader(dataset, **dataloader_kwargs)
        return full_dataloader, None, data_structure
    else:
        if train_indices is None or val_indices is None:
            # Split the dataset into train and validation sets
            train_size = int(train_frac * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        else:
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

        # Create dataloaders with optional sorted domain samplers
        train_dataloader = create_dataloader(train_dataset, **dataloader_kwargs)
        val_dataloader = create_dataloader(val_dataset, **dataloader_kwargs)

        return train_dataloader, val_dataloader, data_structure
