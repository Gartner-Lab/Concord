import numpy as np
from .dataloader import DataLoaderManager # Keep this import

class ChunkLoader:
    """
    An iterator that loads and processes chunks of a large AnnData object,
    often in backed mode, for memory-efficient training.
    
    It yields a tuple of (train_dataloader, val_dataloader, chunk_indices) for each chunk.
    """
    def __init__(self, adata, data_manager: DataLoaderManager, chunk_size: int):
        """
        Initializes the ChunkLoader.

        Args:
            adata (AnnData): The large, potentially backed AnnData object.
            data_manager (DataLoaderManager): A pre-initialized manager responsible for 
                                              turning an AnnData chunk into DataLoaders.
            chunk_size (int): The number of observations (cells) per chunk.
        """
        self.adata = adata
        self.data_manager = data_manager
        self.chunk_size = chunk_size

        self.total_samples = self.adata.n_obs
        self.num_chunks = (self.total_samples + self.chunk_size - 1) // self.chunk_size
        self.indices = np.arange(self.total_samples)

    def __len__(self):
        """Returns the total number of chunks."""
        return self.num_chunks

    def __iter__(self):
        """Shuffles indices at the beginning of each epoch (iteration)."""
        np.random.shuffle(self.indices)
        self.current_chunk_idx = 0
        return self

    def __next__(self):
        """Loads the next chunk into memory and processes it."""
        if self.current_chunk_idx >= self.num_chunks:
            raise StopIteration

        # 1. Define the slice for the current chunk
        start_idx = self.current_chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        chunk_indices = self.indices[start_idx:end_idx]

        # 2. Load ONLY this chunk from disk into memory
        chunk_adata = self.adata[chunk_indices].to_memory()

        # 3. Use the single, persistent DataLoaderManager to process the chunk
        train_loader, val_loader, _ = self.data_manager.anndata_to_dataloader(chunk_adata)
        
        self.current_chunk_idx += 1
        
        # 4. Yield the dataloaders for this chunk and the original indices
        return train_loader, val_loader, chunk_indices
    