import torch
from .sampler import DomainClassSampler
from .anndataset import AnnDataDataset
from torch.utils.data import DataLoader, random_split, Subset

def create_dataloader(dataset, batch_size, sampler_mode, device, drop_last=True):
    if sampler_mode:
        sampler = DomainClassSampler(dataset, batch_size=batch_size, device=device, drop_last=drop_last)
        dataloader = DataLoader(dataset, batch_sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    print(f"Created DataLoader with {len(dataloader)} batches.")
    return dataloader

def anndata_to_dataloader(adata, input_layer_key, domain_key,
                          class_key=None,
                          extra_keys=None,
                          train_frac=0.9,
                          train_indices=None, val_indices=None,
                          batch_size=32,
                          sampler_mode="domain",
                          drop_last=True,
                          keep_indices=True,
                          preprocess=None,
                          device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if preprocess:
        preprocess(adata)

    dataset = AnnDataDataset(adata, input_layer_key=input_layer_key, domain_key=domain_key,
                             class_key=class_key, extra_keys=extra_keys, device=device,
                             keep_indices=keep_indices)

    if train_frac == 1.0:
        # Create a single DataLoader without splitting
        full_dataloader = create_dataloader(dataset, batch_size=batch_size, sampler_mode=sampler_mode,
                                            device=device, drop_last=drop_last)
        return full_dataloader, None
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
        train_dataloader = create_dataloader(train_dataset, batch_size=batch_size,
                                             sampler_mode=sampler_mode, device=device, drop_last=drop_last)
        val_dataloader = create_dataloader(val_dataset, batch_size=batch_size,
                                           sampler_mode=sampler_mode, device=device, drop_last=drop_last)

        return train_dataloader, val_dataloader
