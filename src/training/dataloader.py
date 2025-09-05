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
    """Get the data directory path.
    
    Always returns the full dataset directory.
    
    Args:
        test_mode: Kept for backward compatibility (unused)
        
    Returns:
        Path to the full dataset directory
    """
    base_path = Path("/home/andrea/Desktop/data/tokenized_datasets")
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
    num_subsets: int = 64  # Total number of subsets to divide dataset into
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


def calculate_fixed_eval_dataloader_stats(
    batch_size: int = 32,
    test_mode: bool = False,
    subset_index: int = None,  # 0-based index of subset to load for training (None for all)
    num_subsets: int = 64  # Total number of subsets to divide training files into
) -> Dict:
    """Calculate dataloader statistics for fixed eval mode without loading actual data.
    
    This function calculates stats for:
    - Fixed eval dataset (always tokens_0.npy)
    - Training data from tokens_1.npy onwards based on subset
    
    Args:
        batch_size: Number of sequences per batch
        test_mode: If True, only use tokens_1.npy for training
        subset_index: 0-based index of subset to load for training (None for all)
        num_subsets: Total number of subsets to divide training files into (default 32)
        
    Returns:
        Dictionary with calculated statistics
    """
    # Get data directory and load metadata
    data_dir = _get_data_directory(test_mode)
    metadata = _load_metadata(data_dir)
    
    # Calculate eval dataset size (always tokens_0.npy)
    eval_file = data_dir / "tokens_0.npy"
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    
    eval_shape = np.load(eval_file, mmap_mode='r').shape
    eval_sequences = eval_shape[0]
    eval_batches = (eval_sequences + batch_size - 1) // batch_size  # drop_last=False for eval
    
    # Get training files (all files except tokens_0.npy)
    all_numpy_files = sorted(data_dir.glob("tokens_*.npy"))
    train_numpy_files = [f for f in all_numpy_files if f.name != "tokens_0.npy"]
    
    if not train_numpy_files:
        raise FileNotFoundError(f"No training data files found in {data_dir}")
    
    # Select files for the specified subset (test mode now also uses subset division)
    train_numpy_files = select_files_for_subset(
        train_numpy_files,
        subset_index=subset_index,
        num_subsets=num_subsets,
        verbose=False  # Don't print in stats function
    )
    
    # Calculate total training sequences
    train_sequences = 0
    for file_path in train_numpy_files:
        arr_shape = np.load(file_path, mmap_mode='r').shape
        sequences_in_file = arr_shape[0]
        train_sequences += sequences_in_file
    
    # Calculate batch counts (100% for training, no split)
    train_batches = train_sequences // batch_size  # drop_last=True for train
    
    return {
        'train_size': train_sequences,
        'eval_size': eval_sequences,
        'total_sequences': train_sequences + eval_sequences,
        'train_batches': train_batches,
        'eval_batches': eval_batches,
        'num_train_files': len(train_numpy_files),
        'window_size': metadata['window_size'],
        'vocab_size': metadata['vocab_size'],
        'subset_index': subset_index,
        'num_subsets': num_subsets
    }


def calculate_dataloader_stats(
    batch_size: int = 32,
    val_split: float = 0.1,
    test_mode: bool = False,
    subset_index: int = None,  # 0-based index of subset to load (None for all)
    num_subsets: int = 64  # Total number of subsets to divide dataset into
) -> Dict:
    """Calculate dataloader statistics without loading actual data.
    
    DEPRECATED: Use calculate_fixed_eval_dataloader_stats for fixed eval mode.
    This function is kept for backward compatibility.
    
    Args:
        batch_size: Number of sequences per batch
        val_split: Fraction of data for validation 
        test_mode: Use subset for testing
        subset_index: 0-based index of subset to load (None for all files)
        num_subsets: Total number of subsets to divide dataset into (default 32)
        
    Returns:
        Dictionary with calculated statistics
    """
    # For backward compatibility, redirect to fixed eval stats
    # This approximates the old behavior
    stats = calculate_fixed_eval_dataloader_stats(
        batch_size=batch_size,
        test_mode=test_mode,
        subset_index=subset_index,
        num_subsets=num_subsets
    )
    
    # Approximate old format (with train/val split)
    # Note: This is not exact since we now use fixed eval
    return {
        'train_size': stats['train_size'],
        'val_size': stats['eval_size'],  # Use eval as "val" for compatibility
        'total_sequences': stats['total_sequences'],
        'train_batches': stats['train_batches'],
        'val_batches': stats['eval_batches'],  # Use eval as "val" for compatibility
        'num_files': stats['num_train_files'] + 1,  # Include eval file
        'window_size': stats['window_size'],
        'vocab_size': stats['vocab_size'],
        'subset_index': stats['subset_index'],
        'num_subsets': stats['num_subsets']
    }

def create_fixed_eval_dataloaders(
    batch_size: int = 32,
    test_mode: bool = False,
    verbose: bool = True,
    subset_index: int = None,  # 0-based index of subset to load for training (None for all)
    num_subsets: int = 64  # Total number of subsets to divide training files into
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and eval DataLoaders with fixed evaluation dataset.
    
    This function:
    1. Always loads tokens_0.npy as the evaluation dataset (entire file, no splitting)
    2. Loads specified subset from tokens_1.npy onwards for training (100% training, no val split)
    3. Eval dataset stays on CPU (no pinned memory) for memory efficiency
    4. Training data goes to GPU for zero-copy access
    
    Args:
        batch_size: Number of sequences per batch
        test_mode: If True, only load tokens_1.npy for training (single file)
        verbose: Print progress
        subset_index: 0-based index of subset to load for training (None for all)
        num_subsets: Total number of subsets to divide training files into (default 32)
        
    Returns:
        Tuple of (train_loader, eval_loader, info_dict)
    """
    if verbose:
        print(f"Creating Fixed Eval DataLoaders (FineWeb dataset)")
    
    # Get data directory and load metadata
    data_dir = _get_data_directory(test_mode)
    metadata = _load_metadata(data_dir)
    
    if verbose:
        print(f"\nDataset info:")
        print(f"  Window size: {metadata['window_size']} tokens")
        print(f"  Vocab size: {metadata['vocab_size']}")
        print(f"  Total sequences: {metadata['total_sequences']:,}")
    
    # Step 1: Load fixed evaluation dataset (always tokens_0.npy)
    if verbose:
        print(f"\nLoading fixed evaluation dataset (tokens_0.npy)...")
    
    eval_file = data_dir / "tokens_0.npy"
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    
    eval_sequences = np.load(eval_file)
    if verbose:
        print(f"  Loaded {len(eval_sequences):,} evaluation sequences")
        eval_gb = eval_sequences.nbytes / (1024**3)
        print(f"  Eval data: {eval_gb:.2f} GB")
    
    # Create eval dataset on CPU (no pinned memory for memory efficiency)
    eval_dataset = SimpleDataset(eval_sequences, device='cpu', verbose=False)
    del eval_sequences
    gc.collect()
    
    # Step 2: Load training data (tokens_1.npy onwards)
    if verbose:
        print(f"\nLoading training data...")
    
    # Get all numpy files EXCEPT tokens_0.npy (which is for eval)
    all_numpy_files = sorted(data_dir.glob("tokens_*.npy"))
    # Filter out tokens_0.npy to get only training files
    train_numpy_files = [f for f in all_numpy_files if f.name != "tokens_0.npy"]
    
    if not train_numpy_files:
        raise FileNotFoundError(f"No training data files found in {data_dir}")
    
    if verbose:
        print(f"  Found {len(train_numpy_files)} training files (excluding eval file)")
    
    # Select files for the specified subset (test mode now also uses subset division)
    train_numpy_files = select_files_for_subset(
        train_numpy_files, 
        subset_index=subset_index,
        num_subsets=num_subsets,
        verbose=verbose
    )
    
    if test_mode and verbose:
        print(f"  Test mode: using subset {subset_index+1 if subset_index is not None else 'all'}/{num_subsets} for training")
    
    # Load and concatenate training files
    all_train_arrays = []
    total_train_sequences = 0
    
    for i, file_path in enumerate(train_numpy_files, 1):
        if verbose and i == 1:
            print(f"  Loading {len(train_numpy_files)} training files...", end='', flush=True)
        
        arr = np.load(file_path)
        all_train_arrays.append(arr)
        total_train_sequences += len(arr)
        
        if verbose and i == len(train_numpy_files):
            print(f" done ({total_train_sequences:,} sequences)")
    
    # Concatenate all training arrays
    if verbose:
        print(f"Concatenating {len(all_train_arrays)} training arrays...")
    
    num_files_actually_loaded = len(all_train_arrays)
    train_sequences = np.concatenate(all_train_arrays, axis=0)
    
    # Free memory from individual arrays
    del all_train_arrays
    gc.collect()
    
    if verbose:
        train_numpy_gb = train_sequences.nbytes / (1024**3)
        print(f"Total training data: {train_numpy_gb:.2f} GB")
    
    # Step 3: Create training dataset on GPU (100% for training, no split)
    train_dataset = SimpleDataset(train_sequences, device='cuda', verbose=verbose)
    
    # Free train numpy array
    del train_sequences
    gc.collect()
    
    # Step 4: Create DataLoaders
    
    # Train loader - data already on GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Always shuffle training data
        num_workers=0,  # GPU data doesn't need workers
        pin_memory=False,  # Already on GPU
        drop_last=True
    )
    
    # Eval loader - data on CPU (no pinned memory for memory efficiency)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Will transfer to GPU during validation
        pin_memory=False,  # No pinned memory to save RAM
        drop_last=False
    )
    
    # Report final memory usage
    if verbose:
        train_gb = train_dataset.data.element_size() * train_dataset.data.nelement() / (1024**3)
        eval_gb = eval_dataset.data.element_size() * eval_dataset.data.nelement() / (1024**3)
        
        print(f"Memory: Train {train_gb:.1f}GB GPU | Eval {eval_gb:.1f}GB CPU (no pinning)")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        print(f"Batch size: {batch_size} | Train batches: {len(train_loader):,} | Eval batches: {len(eval_loader):,}")
    
    # Return dataloaders and info
    info = {
        'train_size': len(train_dataset),
        'eval_size': len(eval_dataset), 
        'total_size': len(train_dataset) + len(eval_dataset),
        'train_device': 'cuda',
        'eval_device': 'cpu (no pinning)',
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'metadata': metadata,
        'subset_index': subset_index,
        'num_subsets': num_subsets,
        'num_files_loaded': num_files_actually_loaded,  # Actual number of training files loaded
        'is_fixed_eval': True  # Flag to indicate fixed eval mode
    }
    
    return train_loader, eval_loader, info


def destroy_dataloaders(train_loader, val_loader, info, verbose=True):
    """Properly destroy dataloaders and free GPU/CPU memory.
    
    Args:
        train_loader: Training dataloader to destroy
        val_loader: Validation dataloader to destroy (or eval_loader)
        info: Info dict containing dataset references
        verbose: Print memory cleanup information
    """
    if verbose:
        print("\nDestroying dataloaders and freeing memory...")
    
    # Delete datasets and their data
    # Handle both old names (train_dataset, val_dataset) and new names (eval_dataset)
    for dataset_key in ['train_dataset', 'val_dataset', 'eval_dataset']:
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
