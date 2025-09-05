#!/usr/bin/env python3
"""FineWeb dataset downloader using HuggingFace's native capabilities.

This module downloads the FineWeb dataset from HuggingFace using the built-in
load_dataset function with optimized parameters for full dataset downloading.

Features:
- Native HuggingFace parallel downloading with num_proc
- Automatic caching and resume capabilities
- Efficient JSONL export with chunking
- Support for different dataset configurations (sample-10BT, sample-100BT, full)
- Simple and maintainable codebase
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

# Try to import datasets library
try:
    from datasets import load_dataset, Dataset
except ImportError:
    print("Error: datasets library not found. Please install it with:")
    print("  pip install datasets")
    sys.exit(1)


class FineWebDownloader:
    """Simplified FineWeb dataset downloader using HuggingFace native capabilities."""
    
    # Dataset configurations with their sizes
    DATASET_CONFIGS = {
        "sample-10BT": {
            "name": "sample-10BT",
            "size_gb": 27.6,
            "tokens": "10 billion",
            "description": "10B token sample"
        },
        "sample-100BT": {
            "name": "sample-100BT",
            "size_gb": 277.4,
            "tokens": "100 billion",
            "description": "100B token sample"
        },
        "sample-350BT": {
            "name": "sample-350BT", 
            "size_gb": 968.9,
            "tokens": "350 billion",
            "description": "350B token sample"
        },
        "default": {
            "name": "default",
            "size_gb": 45000,  # ~45TB
            "tokens": "15 trillion",
            "description": "Full FineWeb dataset (15T tokens)"
        }
    }
    
    def __init__(self, 
                 output_dir: Path,
                 config_name: str = "sample-100BT",
                 cache_dir: Optional[Path] = None,
                 num_proc: int = 4):
        """Initialize the FineWeb downloader.
        
        Args:
            output_dir: Directory to save JSONL files
            config_name: Dataset configuration to download
            cache_dir: Cache directory for HuggingFace datasets
            num_proc: Number of processes for parallel downloading
        """
        self.output_dir = output_dir
        self.config_name = config_name
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "datasets"
        self.num_proc = num_proc
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        if config_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Invalid config: {config_name}. Choose from: {list(self.DATASET_CONFIGS.keys())}")
        
        self.config = self.DATASET_CONFIGS[config_name]
    
    def download_dataset(self, max_samples: Optional[int] = None) -> Dataset:
        """Download the FineWeb dataset using HuggingFace's load_dataset.
        
        Args:
            max_samples: Maximum number of samples to download (for testing)
            
        Returns:
            Downloaded dataset
        """
        print(f"\nDownloading FineWeb dataset")
        print("=" * 60)
        print(f"Configuration: {self.config_name}")
        print(f"Description: {self.config['description']}")
        print(f"Size: ~{self.config['size_gb']:.1f} GB")
        print(f"Tokens: {self.config['tokens']}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Parallel processes: {self.num_proc}")
        print("=" * 60)
        
        if self.config['size_gb'] > 1000:
            print("\n⚠️  WARNING: This is a very large dataset!")
            print(f"   Estimated download size: {self.config['size_gb']:.1f} GB")
            print("   Ensure you have sufficient disk space before proceeding.")
            response = input("\nContinue with download? (y/n): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                sys.exit(0)
        
        print("\nDownloading dataset from HuggingFace...")
        print("(This will cache the dataset for future use)")
        
        try:
            # Download the dataset with native HuggingFace capabilities
            dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name=self.config['name'],
                split="train",
                cache_dir=str(self.cache_dir),
                num_proc=self.num_proc if self.num_proc > 1 else None,
                trust_remote_code=True
            )
            
            # Apply sample limit if specified
            if max_samples:
                print(f"\nLimiting to {max_samples:,} samples for testing...")
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            print(f"\n✓ Dataset downloaded successfully!")
            print(f"  Total samples: {len(dataset):,}")
            
            return dataset
            
        except Exception as e:
            print(f"\n❌ Error downloading dataset: {e}")
            sys.exit(1)
    
    def save_to_jsonl(self, 
                      dataset: Dataset, 
                      chunk_size: int = 100000,
                      single_file: bool = False) -> Dict[str, Any]:
        """Save the dataset to JSONL format.
        
        Args:
            dataset: HuggingFace dataset to save
            chunk_size: Number of documents per JSONL file
            single_file: If True, save as a single JSONL file
            
        Returns:
            Dictionary with save statistics
        """
        print(f"\nSaving dataset to JSONL format...")
        print(f"Chunk size: {chunk_size:,} documents per file")
        
        start_time = time.time()
        total_bytes = 0
        file_count = 0
        
        if single_file:
            # Save as single file
            output_path = self.output_dir / "fineweb_complete.jsonl"
            print(f"Saving to: {output_path}")
            
            dataset.to_json(
                str(output_path),
                batch_size=chunk_size,
                num_proc=self.num_proc if self.num_proc > 1 else None
            )
            file_count = 1
            total_bytes = output_path.stat().st_size
            
        else:
            # Save in chunks
            total_samples = len(dataset)
            num_chunks = (total_samples + chunk_size - 1) // chunk_size
            
            print(f"Creating {num_chunks} chunk files...")
            
            with tqdm(total=num_chunks, desc="Saving chunks") as pbar:
                for i in range(0, total_samples, chunk_size):
                    chunk_end = min(i + chunk_size, total_samples)
                    chunk_data = dataset.select(range(i, chunk_end))
                    
                    # Save chunk
                    chunk_filename = f"fineweb_chunk_{file_count:05d}.jsonl"
                    chunk_path = self.output_dir / chunk_filename
                    
                    chunk_data.to_json(
                        str(chunk_path),
                        num_proc=self.num_proc if self.num_proc > 1 else None
                    )
                    
                    total_bytes += chunk_path.stat().st_size
                    file_count += 1
                    pbar.update(1)
        
        # Save metadata
        metadata = {
            'dataset_config': self.config_name,
            'total_samples': len(dataset),
            'total_bytes': total_bytes,
            'chunk_size': chunk_size,
            'num_files': file_count,
            'single_file': single_file,
            'save_time_seconds': time.time() - start_time
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Dataset saved successfully!")
        print(f"  Files created: {file_count}")
        print(f"  Total size: {total_bytes / 1e9:.2f} GB")
        print(f"  Time taken: {metadata['save_time_seconds'] / 60:.1f} minutes")
        
        return metadata




def main():
    """Main entry point for FineWeb dataset downloader."""
    parser = argparse.ArgumentParser(
        description="Download FineWeb dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available configurations:
  sample-10BT   : 10B tokens, ~27.6 GB
  sample-100BT  : 100B tokens, ~277.4 GB (default)
  sample-350BT  : 350B tokens, ~968.9 GB
  default       : Full dataset, 15T tokens, ~45 TB

Examples:
  # Test mode - download 1000 samples
  python -m src.dataset_preparation.fineweb_download --test
  
  # Download sample-10BT dataset
  python -m src.dataset_preparation.fineweb_download --config sample-10BT
  
  # Download with 8 parallel processes
  python -m src.dataset_preparation.fineweb_download --num-proc 8
  
  # Custom output directory
  python -m src.dataset_preparation.fineweb_download --output /path/to/output
"""
    )
    
    parser.add_argument("--test", action="store_true",
                       help="Test mode: download only 1000 samples")
    parser.add_argument("--config", type=str, default="sample-100BT",
                       choices=list(FineWebDownloader.DATASET_CONFIGS.keys()),
                       help="Dataset configuration to download (default: sample-100BT)")
    parser.add_argument("--num-proc", type=int, default=4,
                       help="Number of parallel processes (default: 4)")
    parser.add_argument("--chunk-size", type=int, default=100000,
                       help="Documents per chunk file (default: 100000)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to download (default: all)")
    parser.add_argument("--single-file", action="store_true",
                       help="Save as single JSONL file instead of chunks")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: data/raw/fineweb)")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="HuggingFace cache directory (default: ~/.cache/huggingface)")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = Path(args.output) if args.output else Path("data/raw/fineweb")
    
    # Set cache directory
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    
    # Handle test mode
    max_samples = args.max_samples
    if args.test:
        max_samples = 1000
        print("Running in test mode: limiting to 1000 samples")
    
    # Create downloader
    downloader = FineWebDownloader(
        output_dir=output_dir,
        config_name=args.config,
        cache_dir=cache_dir,
        num_proc=args.num_proc
    )
    
    try:
        # Download dataset
        dataset = downloader.download_dataset(max_samples=max_samples)
        
        # Save to JSONL
        metadata = downloader.save_to_jsonl(
            dataset,
            chunk_size=args.chunk_size,
            single_file=args.single_file
        )
        
        print("\n" + "=" * 60)
        print("✓ Download and save complete!")
        print("=" * 60)
        print(f"Configuration: {args.config}")
        print(f"Total samples: {metadata['total_samples']:,}")
        print(f"Total size: {metadata['total_bytes'] / 1e9:.2f} GB")
        print(f"Files created: {metadata['num_files']}")
        print(f"Output directory: {output_dir}")
        print(f"Metadata saved to: {output_dir / 'metadata.json'}")
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()