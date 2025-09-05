#!/usr/bin/env python3
"""Memory-efficient PyTorch DataLoader for language model training.

This module provides a simplified, memory-efficient implementation that:
- Loads numpy files and splits BEFORE tensor conversion
- Avoids memory duplication from PyTorch's advanced indexing  
"""

import gc
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import orjson


def _get_data_directory(test_mode: bool = False) -> Path:
    """Get the appropriate data directory based on mode.
    
    Args:
        test_mode: Whether to use test dataset
        
    Returns:
        Path to the data directory
    """
    base_path = Path("/home/andrea/Desktop/data/tokenized_datasets")
    
    if test_mode:
        data_dir = base_path / "fineweb_test_dataset"
        if not data_dir.exists():
            data_dir = base_path / "fineweb_full_dataset"
    else:
        data_dir = base_path / "fineweb_full_dataset"
    
    return data_dir


def _load_metadata(data_dir: Path) -> Dict:
    """Load metadata from the data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary containing metadata
    """
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        return orjson.loads(f.read())

def get_file_subset_indices(total_files: int, subset_index: int, num_subsets: int) -> Tuple[int, int]:
    """Calculate file indices for a given subset using fair distribution.
    
    Distributes files as evenly as possible across subsets. If files don't divide
    evenly, earlier subsets get one extra file each until remainder is distributed.
    
    Args:
        total_files: Total number of files available
        subset_index: 0-based index of the subset to get
        num_subsets: Total number of subsets to divide into
        
    Returns:
        Tuple of (start_idx, end_idx) for array slicing
    """
    if not 0 <= subset_index < num_subsets:
        raise ValueError(f"subset_index {subset_index} must be in range [0, {num_subsets-1}]")
    
    base_size = total_files // num_subsets
    remainder = total_files % num_subsets
    
    # Subsets 0 to remainder-1 get one extra file each
    if subset_index < remainder:
        start = subset_index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (subset_index - remainder) * base_size
        end = start + base_size
    
    return start, end


def select_files_for_subset(
    numpy_files: list, 
    subset_index: int = None,
    num_subsets: int = 32,
    verbose: bool = False
) -> list:
    """Select files for the specified subset with flexible division.
    
    Args:
        numpy_files: List of all available numpy file paths
        subset_index: 0-based subset index (None for all files)
        num_subsets: Total number of subsets to divide into
        verbose: Print selection details
        
    Returns:
        List of selected file paths for the subset
    """
    total_files = len(numpy_files)
    
    if subset_index is None:
        # Load all files
        if verbose:
            print(f"  Loading all {total_files} files")
        return numpy_files
    
    if not 0 <= subset_index < num_subsets:
        raise ValueError(f"subset_index {subset_index} must be in range [0, {num_subsets-1}]")
    
    start_idx, end_idx = get_file_subset_indices(total_files, subset_index, num_subsets)
    selected_files = numpy_files[start_idx:end_idx]
    
    if verbose:
        print(f"  Loading subset {subset_index+1}/{num_subsets}: {len(selected_files)} of {total_files} files "
              f"(indices {start_idx}-{end_idx-1})")
    
    return selected_files



class SimpleDataset(Dataset):
    """Simple dataset wrapper for pre-loaded tensor data.
    
    Minimal implementation that just holds a tensor and returns slices.
    No complex memory management or sharing."""
    
    def __init__(self, sequences_numpy, device='cpu', verbose=True):
        """Initialize dataset from numpy array.
        
        Args:
            sequences_numpy: Numpy array of shape (num_sequences, window_size)
            device: Device to place tensor on ('cuda' or 'cpu')
            verbose: Print loading stats
        """
        self.device = device
        self.verbose = verbose
        
        if verbose:
            print(f"Creating dataset on {device}...")
        
        # Convert to tensor with int16 for storage efficiency
        # We'll convert to int64 on-the-fly in __getitem__
        # Explicitly specify dtype to prevent any automatic promotion
        self.data = torch.tensor(sequences_numpy, dtype=torch.int16)
        
        # Delete the numpy array to free memory immediately
        del sequences_numpy
        gc.collect()
        
        # Move to device
        if device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            self.data = self.data.to('cuda')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert from int16 storage to int64 for processing
        return self.data[idx].to(torch.long)


def create_simple_train_val_dataloaders(
    batch_size: int = 32,
    val_split: float = 0.1,
    shuffle_train: bool = True,
    test_mode: bool = False,
    verbose: bool = True,
    seed: int = 42,
    subset_index: int = None,  # 0-based index of subset to load (None for all)
    num_subsets: int = 32  # Total number of subsets to divide dataset into
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and validation DataLoaders with simplified memory-efficient approach.
    
    This function:
    1. Loads specified subset of numpy files and concatenates them
    2. Splits the numpy array (90/10) BEFORE tensor conversion
    3. Creates two independent datasets (train on GPU, val on CPU)
    4. Returns two independent dataloaders
    
    This avoids memory duplication from PyTorch's advanced indexing.
    
    Args:
        batch_size: Number of sequences per batch
        val_split: Fraction of data for validation (default 0.1)
        shuffle_train: Whether to shuffle training data
        test_mode: Use subset for testing
        verbose: Print progress
        seed: Random seed for splitting
        subset_index: 0-based index of subset to load (None for all files)
        num_subsets: Total number of subsets to divide dataset into (default 32)
        
    Returns:
        Tuple of (train_loader, val_loader, info_dict)
    """
    if verbose:
        print(f"Creating Simple Train/Val DataLoaders (FineWeb dataset)")
    
    # Get data directory and load metadata
    data_dir = _get_data_directory(test_mode)
    metadata = _load_metadata(data_dir)
    
    if verbose:
        print(f"\nDataset info:")
        print(f"  Window size: {metadata['window_size']} tokens")
        print(f"  Vocab size: {metadata['vocab_size']}")
        print(f"  Total sequences: {metadata['total_sequences']:,}")
    
    # Step 1: Load and concatenate specified subset of numpy files
    if verbose:
        subset_desc = "all" if subset_index is None else f"subset {subset_index+1}/{num_subsets}"
        print(f"Loading numpy files from disk ({subset_desc})...")
    
    numpy_files = sorted(data_dir.glob("tokens_*.npy"))
    if not numpy_files:
        raise FileNotFoundError(f"No tokenized data files found in {data_dir}")
    
    # Use the new helper function to select files
    numpy_files = select_files_for_subset(
        numpy_files, 
        subset_index=subset_index,
        num_subsets=num_subsets,
        verbose=verbose
    )
    
    # Load selected files and concatenate
    all_arrays = []
    total_sequences = 0
    
    for i, file_path in enumerate(numpy_files, 1):
        if verbose and i == 1:
            print(f"  Loading {len(numpy_files)} files...", end='', flush=True)
        
        arr = np.load(file_path)
        all_arrays.append(arr)
        total_sequences += len(arr)
        
        if verbose and i == len(numpy_files):
            print(f" done ({total_sequences:,} sequences)")
        
        # For test mode, limit to first few files
        if test_mode and total_sequences >= 10000:
            if verbose:
                print(f"  Test mode: Stopping at {total_sequences:,} sequences")
            break
    
    # Concatenate all arrays
    if verbose:
        print(f"Concatenating {len(all_arrays)} arrays...")
    
    num_files_actually_loaded = len(all_arrays)  # Save count before deletion
    all_sequences = np.concatenate(all_arrays, axis=0)
    
    # Free memory from individual arrays
    del all_arrays
    gc.collect()
    
    if verbose:
        numpy_gb = all_sequences.nbytes / (1024**3)
        print(f"Total data: {numpy_gb:.2f} GB")
    
    # Step 2: Shuffle and split BEFORE tensor conversion
    if verbose:
        print(f"Splitting data (train={100*(1-val_split):.0f}%, val={100*val_split:.0f}%)...")
    
    # Create random permutation
    np.random.seed(seed)
    indices = np.random.permutation(len(all_sequences))
    
    # Calculate split point
    val_size = int(len(all_sequences) * val_split)
    train_size = len(all_sequences) - val_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Get train and val data (these are views if indices are sorted, copies otherwise)
    # But since we're permuting, these will be copies - still more efficient than PyTorch indexing
    train_sequences = all_sequences[train_indices]
    val_sequences = all_sequences[val_indices]
    
    if verbose:
        print(f"  Split: Train {len(train_sequences):,} | Val {len(val_sequences):,}")
    
    # Free the full array
    del all_sequences, indices, train_indices, val_indices
    gc.collect()
    
    # Step 3: Create independent datasets
    
    # Train dataset on GPU
    train_dataset = SimpleDataset(train_sequences, device='cuda', verbose=verbose)
    
    # Free train numpy array
    del train_sequences
    gc.collect()
    
    # Val dataset on CPU (with pinned memory)
    val_dataset = SimpleDataset(val_sequences, device='cpu', verbose=verbose)
    
    # Free val numpy array  
    del val_sequences
    gc.collect()
    
    # Step 4: Create DataLoaders
    
    # Train loader - data already on GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0,  # GPU data doesn't need workers
        pin_memory=False,  # Already on GPU
        drop_last=True
    )
    
    # Val loader - data on CPU with pinned memory
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Use workers for CPU->GPU transfer
        pin_memory=False,  # Already pinned in dataset
        persistent_workers=True,
        drop_last=False
    )
    
    # Report final memory usage
    if verbose:

        train_gb = train_dataset.data.element_size() * train_dataset.data.nelement() / (1024**3)
        val_gb = val_dataset.data.element_size() * val_dataset.data.nelement() / (1024**3)
        
        print(f"Memory: Train {train_gb:.1f}GB GPU | Val {val_gb:.1f}GB CPU")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        print(f"Batch size: {batch_size} | Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")
    
    # Return dataloaders and info
    info = {
        'train_size': train_size,
        'val_size': val_size, 
        'total_size': train_size + val_size,
        'train_device': 'cuda',
        'val_device': 'cpu (pinned)',
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'metadata': metadata,
        'subset_index': subset_index,
        'num_subsets': num_subsets,
        'num_files_loaded': num_files_actually_loaded  # Actual number of files loaded
    }
    
    return train_loader, val_loader, info


def calculate_dataloader_stats(
    batch_size: int = 32,
    val_split: float = 0.1,
    test_mode: bool = False,
    subset_index: int = None,  # 0-based index of subset to load (None for all)
    num_subsets: int = 32  # Total number of subsets to divide dataset into
) -> Dict:
    """Calculate dataloader statistics without loading actual data.
    
    This function reads metadata and file sizes to compute batch counts
    without the memory overhead of loading the actual token data.
    
    Args:
        batch_size: Number of sequences per batch
        val_split: Fraction of data for validation 
        test_mode: Use subset for testing
        subset_index: 0-based index of subset to load (None for all files)
        num_subsets: Total number of subsets to divide dataset into (default 32)
        
    Returns:
        Dictionary with calculated statistics
    """
    # Get data directory and load metadata
    data_dir = _get_data_directory(test_mode)
    metadata = _load_metadata(data_dir)
    
    # Get list of numpy files
    numpy_files = sorted(data_dir.glob("tokens_*.npy"))
    if not numpy_files:
        raise FileNotFoundError(f"No tokenized data files found in {data_dir}")
    
    # Use the new helper function to select files
    numpy_files = select_files_for_subset(
        numpy_files,
        subset_index=subset_index,
        num_subsets=num_subsets,
        verbose=False  # Don't print in stats function
    )
    
    # Calculate total sequences by checking file shapes
    # We can get this from the metadata or by quickly checking file shapes
    total_sequences = 0
    for file_path in numpy_files:
        # Use numpy's memmap to read just the header and get shape without loading data
        arr_shape = np.load(file_path, mmap_mode='r').shape
        sequences_in_file = arr_shape[0]
        total_sequences += sequences_in_file
        
        # For test mode, limit sequences
        if test_mode and total_sequences >= 10000:
            total_sequences = min(total_sequences, 10000)
            break
    
    # Calculate train/val split
    val_size = int(total_sequences * val_split)
    train_size = total_sequences - val_size
    
    # Calculate batch counts
    train_batches = train_size // batch_size  # drop_last=True for train
    val_batches = (val_size + batch_size - 1) // batch_size  # drop_last=False for val
    
    return {
        'train_size': train_size,
        'val_size': val_size,
        'total_sequences': total_sequences,
        'train_batches': train_batches,
        'val_batches': val_batches,
        'num_files': len(numpy_files),
        'window_size': metadata['window_size'],
        'vocab_size': metadata['vocab_size'],
        'subset_index': subset_index,
        'num_subsets': num_subsets
    }

def destroy_dataloaders(train_loader, val_loader, info, verbose=True):
    """Properly destroy dataloaders and free GPU/CPU memory.
    
    Args:
        train_loader: Training dataloader to destroy
        val_loader: Validation dataloader to destroy
        info: Info dict containing dataset references
        verbose: Print memory cleanup information
    """
    if verbose:
        print("\nDestroying dataloaders and freeing memory...")
    
    # Delete datasets and their data
    for dataset_key in ['train_dataset', 'val_dataset']:
        if dataset_key in info:
            if hasattr(info[dataset_key], 'data'):
                del info[dataset_key].data
            del info[dataset_key]
    
    # Delete dataloaders and info
    del train_loader
    del val_loader
    del info
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
