#!/usr/bin/env python3
"""Memory-efficient PyTorch DataLoader for Wikipedia language model training.

This module provides a simplified, memory-efficient implementation that:
- Loads numpy files and splits BEFORE tensor conversion
- Avoids memory duplication from PyTorch's advanced indexing  
- Creates independent train (GPU) and validation (CPU) datasets
- Memory usage: ~22GB (20GB GPU + 2GB CPU) instead of 60GB+
"""

import sys
import time
import gc
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import orjson
try:
    import psutil
except ImportError:
    psutil = None
    print("Warning: psutil not installed, memory reporting will be limited")


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
        elif device == 'cpu' and torch.cuda.is_available():
            # Pin memory for faster GPU transfers during validation
            self.data = self.data.pin_memory()
        
        if verbose:
            pass  # Dataset created silently
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert from int16 storage to int64 for processing
        return self.data[idx].to(torch.long)
    
    def get_stats(self):
        return {
            'num_sequences': len(self),
            'window_size': self.data.shape[1],
            'device': str(self.data.device),
            'dtype': str(self.data.dtype),
            'memory_gb': self.data.element_size() * self.data.nelement() / (1024**3)
        }


def create_simple_train_val_dataloaders(
    batch_size: int = 32,
    val_split: float = 0.1,
    shuffle_train: bool = True,
    test_mode: bool = False,
    verbose: bool = True,
    seed: int = 42,
    file_subset: str = 'all',
    dataset_source: str = 'wikipedia'
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
        file_subset: Which files to load ('all', 'first_fourth', 'second_fourth', 
                     'third_fourth', 'fourth_fourth')
        dataset_source: Which dataset to use ('wikipedia' or 'fineweb')
        
    Returns:
        Tuple of (train_loader, val_loader, info_dict)
    """
    import gc
    
    if verbose:
        print(f"Creating Simple Train/Val DataLoaders (source: {dataset_source})")
    
    # Determine data directory based on dataset source
    base_path = Path("/home/andrea/Desktop/data/tokenized_datasets")
    
    if dataset_source == 'wikipedia':
        if test_mode:
            data_dir = base_path / "codellama_test_dataset"
            if not data_dir.exists():
                data_dir = base_path / "codellama_full_dataset"
        else:
            data_dir = base_path / "codellama_full_dataset"
    elif dataset_source == 'fineweb':
        if test_mode:
            data_dir = base_path / "fineweb_test_dataset"
            if not data_dir.exists():
                data_dir = base_path / "fineweb_full_dataset"
        else:
            data_dir = base_path / "fineweb_full_dataset"
    else:
        raise ValueError(f"Invalid dataset_source: {dataset_source}. Must be 'wikipedia' or 'fineweb'")
    
    # Load metadata
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = orjson.loads(f.read())
    
    if verbose:
        print(f"\nDataset info:")
        print(f"  Window size: {metadata['window_size']} tokens")
        print(f"  Vocab size: {metadata['vocab_size']}")
        print(f"  Total sequences: {metadata['total_sequences']:,}")
    
    # Step 1: Load and concatenate specified subset of numpy files
    if verbose:
        print(f"Loading numpy files from disk (subset: {file_subset})...")
    
    numpy_files = sorted(data_dir.glob("tokens_*.npy"))
    if not numpy_files:
        raise FileNotFoundError(f"No tokenized data files found in {data_dir}")
    
    # Determine which files to load based on file_subset parameter
    total_files = len(numpy_files)
    
    if file_subset == 'first_fourth':
        # Load first fourth of files (files 0-9 for 38 total)
        num_files_per_fourth = total_files // 4
        # First fourth gets one extra if remainder > 0
        if total_files % 4 > 0:
            num_files_per_fourth += 1
        numpy_files = numpy_files[:num_files_per_fourth]
        if verbose:
            print(f"  Loading first fourth: {len(numpy_files)} of {total_files} files (indices 0-{len(numpy_files)-1})")
    elif file_subset == 'second_fourth':
        # Load second fourth of files (files 10-19 for 38 total)
        num_files_per_fourth = total_files // 4
        # Calculate sizes for each fourth
        first_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 0 else 0)
        second_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 1 else 0)
        start_idx = first_fourth_size
        end_idx = first_fourth_size + second_fourth_size
        numpy_files = numpy_files[start_idx:end_idx]
        if verbose:
            print(f"  Loading second fourth: {len(numpy_files)} of {total_files} files (indices {start_idx}-{end_idx-1})")
    elif file_subset == 'third_fourth':
        # Load third fourth of files (files 20-28 for 38 total)
        num_files_per_fourth = total_files // 4
        # Calculate sizes for each fourth
        first_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 0 else 0)
        second_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 1 else 0)
        third_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 2 else 0)
        start_idx = first_fourth_size + second_fourth_size
        end_idx = first_fourth_size + second_fourth_size + third_fourth_size
        numpy_files = numpy_files[start_idx:end_idx]
        if verbose:
            print(f"  Loading third fourth: {len(numpy_files)} of {total_files} files (indices {start_idx}-{end_idx-1})")
    elif file_subset == 'fourth_fourth':
        # Load fourth fourth of files (files 29-37 for 38 total)
        num_files_per_fourth = total_files // 4
        # Calculate where fourth fourth starts
        first_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 0 else 0)
        second_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 1 else 0)
        third_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 2 else 0)
        start_idx = first_fourth_size + second_fourth_size + third_fourth_size
        numpy_files = numpy_files[start_idx:]
        if verbose:
            print(f"  Loading fourth fourth: {len(numpy_files)} of {total_files} files (indices {start_idx}-{total_files-1})")
    elif file_subset == 'all':
        # Load all files (default behavior)
        if verbose:
            print(f"  Loading all {len(numpy_files)} files")
    else:
        raise ValueError(f"Invalid file_subset: {file_subset}. Must be 'all', 'first_fourth', 'second_fourth', 'third_fourth', or 'fourth_fourth'")
    
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
        'file_subset': file_subset,
        'num_files_loaded': num_files_actually_loaded  # Actual number of files loaded
    }
    
    return train_loader, val_loader, info


def calculate_dataloader_stats(
    batch_size: int = 32,
    val_split: float = 0.1,
    test_mode: bool = False,
    file_subset: str = 'all',
    dataset_source: str = 'wikipedia'
) -> Dict:
    """Calculate dataloader statistics without loading actual data.
    
    This function reads metadata and file sizes to compute batch counts
    without the memory overhead of loading the actual token data.
    
    Args:
        batch_size: Number of sequences per batch
        val_split: Fraction of data for validation 
        test_mode: Use subset for testing
        file_subset: Which files to load ('all', 'first_fourth', 'second_fourth',
                     'third_fourth', 'fourth_fourth')
        dataset_source: Which dataset to use ('wikipedia' or 'fineweb')
        
    Returns:
        Dictionary with calculated statistics
    """
    # Determine data directory based on dataset source
    base_path = Path("/home/andrea/Desktop/data/tokenized_datasets")
    
    if dataset_source == 'wikipedia':
        if test_mode:
            data_dir = base_path / "codellama_test_dataset"
            if not data_dir.exists():
                data_dir = base_path / "codellama_full_dataset"
        else:
            data_dir = base_path / "codellama_full_dataset"
    elif dataset_source == 'fineweb':
        if test_mode:
            data_dir = base_path / "fineweb_test_dataset"
            if not data_dir.exists():
                data_dir = base_path / "fineweb_full_dataset"
        else:
            data_dir = base_path / "fineweb_full_dataset"
    else:
        raise ValueError(f"Invalid dataset_source: {dataset_source}. Must be 'wikipedia' or 'fineweb'")
    
    # Load metadata
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = orjson.loads(f.read())
    
    # Get list of numpy files
    numpy_files = sorted(data_dir.glob("tokens_*.npy"))
    if not numpy_files:
        raise FileNotFoundError(f"No tokenized data files found in {data_dir}")
    
    # Determine which files based on file_subset
    total_files = len(numpy_files)
    
    if file_subset == 'first_fourth':
        num_files_per_fourth = total_files // 4
        if total_files % 4 > 0:
            num_files_per_fourth += 1
        numpy_files = numpy_files[:num_files_per_fourth]
    elif file_subset == 'second_fourth':
        num_files_per_fourth = total_files // 4
        first_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 0 else 0)
        second_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 1 else 0)
        start_idx = first_fourth_size
        end_idx = first_fourth_size + second_fourth_size
        numpy_files = numpy_files[start_idx:end_idx]
    elif file_subset == 'third_fourth':
        num_files_per_fourth = total_files // 4
        first_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 0 else 0)
        second_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 1 else 0)
        third_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 2 else 0)
        start_idx = first_fourth_size + second_fourth_size
        end_idx = first_fourth_size + second_fourth_size + third_fourth_size
        numpy_files = numpy_files[start_idx:end_idx]
    elif file_subset == 'fourth_fourth':
        num_files_per_fourth = total_files // 4
        first_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 0 else 0)
        second_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 1 else 0)
        third_fourth_size = num_files_per_fourth + (1 if total_files % 4 > 2 else 0)
        start_idx = first_fourth_size + second_fourth_size + third_fourth_size
        numpy_files = numpy_files[start_idx:]
    elif file_subset != 'all':
        raise ValueError(f"Invalid file_subset: {file_subset}. Must be 'all', 'first_fourth', 'second_fourth', 'third_fourth', or 'fourth_fourth'")
    
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
        'file_subset': file_subset
    }


def get_memory_usage():
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage info for GPU and system
    """
    memory_stats = {}
    
    # GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        memory_stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        memory_stats['gpu_free_gb'] = (torch.cuda.get_device_properties(0).total_memory - 
                                       torch.cuda.memory_reserved()) / (1024**3)
    
    # System memory (only if psutil is available)
    if psutil is not None:
        vm = psutil.virtual_memory()
        memory_stats['ram_used_gb'] = (vm.total - vm.available) / (1024**3)
        memory_stats['ram_available_gb'] = vm.available / (1024**3)
        memory_stats['ram_percent'] = vm.percent
    
    return memory_stats


def destroy_dataloaders(train_loader, val_loader, info, verbose=True):
    """Properly destroy dataloaders and free GPU/CPU memory.
    
    Args:
        train_loader: Training dataloader to destroy
        val_loader: Validation dataloader to destroy
        info: Info dict containing dataset references
        verbose: Print memory cleanup information
    """
    import gc
    
    if verbose:
        print("\nDestroying dataloaders and freeing memory...")
        before_stats = get_memory_usage()
        print(f"  Memory before cleanup:")
        if 'gpu_allocated_gb' in before_stats:
            print(f"    GPU: {before_stats['gpu_allocated_gb']:.2f} GB allocated, "
                  f"{before_stats['gpu_reserved_gb']:.2f} GB reserved")
        if 'ram_used_gb' in before_stats:
            print(f"    RAM: {before_stats['ram_used_gb']:.2f} GB used, "
                  f"{before_stats['ram_available_gb']:.2f} GB available")
    
    # Delete dataset tensors
    if 'train_dataset' in info and hasattr(info['train_dataset'], 'data'):
        del info['train_dataset'].data
    if 'val_dataset' in info and hasattr(info['val_dataset'], 'data'):
        del info['val_dataset'].data
    
    # Delete datasets
    if 'train_dataset' in info:
        del info['train_dataset']
    if 'val_dataset' in info:
        del info['val_dataset']
    
    # Delete dataloaders
    del train_loader
    del val_loader
    del info
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if verbose:
        after_stats = get_memory_usage()
        print(f"  Memory after cleanup:")
        if 'gpu_allocated_gb' in after_stats:
            print(f"    GPU: {after_stats['gpu_allocated_gb']:.2f} GB allocated, "
                  f"{after_stats['gpu_reserved_gb']:.2f} GB reserved")
            if 'gpu_allocated_gb' in before_stats:
                gpu_freed = before_stats['gpu_allocated_gb'] - after_stats['gpu_allocated_gb']
                print(f"    GPU memory freed: {gpu_freed:.2f} GB")
        if 'ram_used_gb' in after_stats:
            print(f"    RAM: {after_stats['ram_used_gb']:.2f} GB used, "
                  f"{after_stats['ram_available_gb']:.2f} GB available")
            if 'ram_used_gb' in before_stats:
                ram_freed = before_stats['ram_used_gb'] - after_stats['ram_used_gb']
                print(f"    RAM freed: {ram_freed:.2f} GB")


def main():
    """Test the simplified memory-efficient train/val dataloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wikipedia DataLoader Test")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for testing")
    parser.add_argument("--test-mode", action="store_true",
                       help="Use test dataset for faster testing")
    
    args = parser.parse_args()
    
    print("Testing Simplified Memory-Efficient Wikipedia DataLoader...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This dataloader requires a GPU.")
        sys.exit(1)
    
    # Create train and val dataloaders with simplified approach
    train_loader, val_loader, info = create_simple_train_val_dataloaders(
        batch_size=args.batch_size,
        val_split=0.1,
        shuffle_train=True,
        test_mode=args.test_mode,
        verbose=True
    )
    
    # Test loading a few batches from training
    print(f"\n{'='*60}")
    print("Testing TRAINING batch loading (GPU-resident)...")
    print(f"{'='*60}")
    
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Test first 3 batches
            break
        
        print(f"\nTrain Batch {i+1}:")
        print(f"  Shape: {batch.shape}")
        print(f"  Device: {batch.device}")
        print(f"  Dtype: {batch.dtype} (converted from int16 storage)")
        print(f"  Min value: {batch.min().item()}")
        print(f"  Max value: {batch.max().item()}")
        print(f"  Sample tokens: {batch[0, :10].tolist()}")
    
    # Test loading a few batches from validation
    print(f"\n{'='*60}")
    print("Testing VALIDATION batch loading (CPU→GPU transfer)...")
    print(f"{'='*60}")
    
    for i, batch in enumerate(val_loader):
        if i >= 3:  # Test first 3 batches
            break
        
        # Transfer to GPU (simulating what happens during validation)
        batch = batch.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        
        print(f"\nVal Batch {i+1}:")
        print(f"  Shape: {batch.shape}")
        print(f"  Device: {batch.device}")
        print(f"  Dtype: {batch.dtype}")
        print(f"  Min value: {batch.min().item()}")
        print(f"  Max value: {batch.max().item()}")
        print(f"  Sample tokens: {batch[0, :10].tolist()}")
    
    # Measure iteration speed for training
    print(f"\n{'='*60}")
    print("Measuring TRAINING iteration speed (100 batches)...")
    
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 100:
            break
        # Simulate minimal processing
        _ = batch.sum()
    
    elapsed = time.time() - start_time
    batches_per_sec = 100 / elapsed if elapsed > 0 else 0
    
    print(f"Train iteration speed: {batches_per_sec:.1f} batches/sec")
    print(f"Time per batch: {elapsed/100*1000:.1f} ms")
    
    # Measure iteration speed for validation
    print(f"\n{'='*60}")
    print("Measuring VALIDATION iteration speed (50 batches)...")
    
    start_time = time.time()
    for i, batch in enumerate(val_loader):
        if i >= 50:
            break
        # Transfer to GPU and process
        batch = batch.to('cuda', non_blocking=True)
        _ = batch.sum()
    
    elapsed = time.time() - start_time
    batches_per_sec = 50 / elapsed if elapsed > 0 else 0
    
    print(f"Val iteration speed: {batches_per_sec:.1f} batches/sec")
    print(f"Time per batch: {elapsed/50*1000:.1f} ms")
    
    print(f"\n{'='*60}")
    print("DataLoader test complete!")
    print(f"\nSummary:")
    print(f"  • Single data load (no duplication)")
    print(f"  • Train: {info['train_size']:,} sequences on {info['train_device']}")
    print(f"  • Val: {info['val_size']:,} sequences on {info['val_device']}")
    print(f"  • Memory optimized: ~50% reduction vs duplicate loading")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)