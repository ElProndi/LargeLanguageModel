#!/usr/bin/env python3
"""Dataset preparation pipeline for tokenizing Wikipedia articles."""

import sys
import time
import argparse
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Any
import numpy as np
import orjson
from numpy.lib.stride_tricks import as_strided
from tokenizer import WikipediaTokenizer


class DatasetTokenizer:
    """Tokenizes cleaned Wikipedia articles for training."""
    
    def __init__(self, window_size: int = 512):
        """Initialize dataset tokenizer.
        
        Args:
            window_size: Context window size in tokens (default 512)
        """
        self.window_size = window_size
        self.tokenizer = None
        self.vocab_size = None
        
        # Statistics tracking
        self.total_articles = 0
        self.total_tokens = 0
        self.total_sequences = 0
        
    def load_tokenizer(self, test_mode: bool = False):
        """Load CodeLlama tokenizer.
        
        Args:
            test_mode: Ignored - always uses full CodeLlama tokenizer
        """
        tokenizer_path = Path("tokenizers/codellama_tokenizer")
        
        # Create WikipediaTokenizer instance (wrapper for CodeLlama)
        self.tokenizer = WikipediaTokenizer()
        
        # Try to load saved tokenizer first, otherwise download
        if tokenizer_path.exists():
            print(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer.load(str(tokenizer_path))
        else:
            print("Downloading CodeLlama tokenizer from HuggingFace...")
            self.tokenizer.train()  # This actually downloads the pre-trained tokenizer
            self.tokenizer.save(str(tokenizer_path))
        
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        print(f"Loaded CodeLlama tokenizer")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def calculate_num_sequences(self, token_lengths: List[int]) -> int:
        """Calculate total number of sequences that will be generated.
        
        Args:
            token_lengths: List of token counts for each article
            
        Returns:
            Total number of sequences that will be created
        """
        stride = self.window_size // 2
        total = 0
        
        for length in token_lengths:
            if length >= self.window_size:
                # Formula: (length - window_size) // stride + 1
                n_windows = (length - self.window_size) // stride + 1
                total += n_windows
        
        return total
    
    def read_all_articles(self, file_path: Path, max_articles: int = None) -> List[str]:
        """Read all articles from file into memory.
        
        Args:
            file_path: Path to cleaned articles file
            max_articles: Maximum number of articles to read (for testing)
            
        Returns:
            List of all article texts
        """
        articles = []
        try:
            print(f"  Loading {file_path.name} into memory...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        articles.append(line)
                        
                        # Stop if we've read enough articles (for testing)
                        if max_articles and len(articles) >= max_articles:
                            break
            
            print(f"  Loaded {len(articles):,} articles (~{sum(len(a) for a in articles) / (1024*1024):.1f} MB)")
            return articles
                    
        except IOError as e:
            raise RuntimeError(f"Failed to read {file_path}: {e}")
    
    def create_sliding_windows(self, token_ids: np.ndarray) -> np.ndarray:
        """Create sliding window sequences using NumPy stride tricks for maximum performance.
        
        Args:
            token_ids: NumPy array of token IDs from an article
            
        Returns:
            2D NumPy array of shape (num_windows, window_size)
        """
        n_tokens = len(token_ids)
        
        # Return empty array if article is too short
        if n_tokens < self.window_size:
            return np.array([], dtype=np.int16).reshape(0, self.window_size)
        
        # Calculate stride and number of windows
        stride = self.window_size // 2
        n_windows = (n_tokens - self.window_size) // stride + 1
        
        # Use stride tricks to create windows without copying data
        # This creates a "view" into the original array
        windows = as_strided(
            token_ids,
            shape=(n_windows, self.window_size),
            strides=(token_ids.strides[0] * stride, token_ids.strides[0])
        )
        
        # Return contiguous copy for efficient saving
        return np.ascontiguousarray(windows, dtype=np.int16)
    
    def process_file(self, file_path: Path, output_path: Path, test_mode: bool = False) -> Dict[str, Any]:
        """Process a single cleaned file and save tokenized sequences.
        
        Args:
            file_path: Path to cleaned articles file
            output_path: Path to save tokenized sequences
            test_mode: If True, only process 10000 articles for testing
            
        Returns:
            Statistics dictionary for this file
        """
        print(f"\nProcessing: {file_path.name}")
        if test_mode:
            print(f"Test mode: Processing first 10,000 articles only")
        print(f"{'='*60}")
        
        file_articles = 0
        file_tokens = 0
        file_sequences = 0
        
        start_time = time.time()
        
        # Load all articles into memory
        max_articles = 10000 if test_mode else None
        all_articles = self.read_all_articles(file_path, max_articles)
        
        # IMPORTANT: All articles are sent to the tokenizer at once for maximum efficiency.
        # The CodeLlama tokenizer handles internal batching and parallelization.
        print(f"  Sending {len(all_articles):,} articles to tokenizer (all at once)...")
        # WikipediaTokenizer.encode handles batch encoding internally
        encoded_batch = self.tokenizer.encode(all_articles, add_special_tokens=True)
        # Convert to list of objects with 'ids' attribute for compatibility
        encodings = [type('Encoding', (), {'ids': ids})() for ids in encoded_batch]
        
        # PASS 1: Calculate total sequences needed for pre-allocation
        print(f"  Calculating total sequences for pre-allocation...")
        token_lengths = [len(encoding.ids) for encoding in encodings]
        total_sequences = self.calculate_num_sequences(token_lengths)
        
        if total_sequences == 0:
            print(f"  Warning: No sequences generated from {file_path.name}")
            return {
                "file": file_path.name,
                "articles": len(all_articles),
                "tokens": sum(token_lengths),
                "sequences": 0,
                "processing_time": time.time() - start_time,
                "articles_per_sec": 0
            }
        
        # Pre-allocate the entire sequences array
        print(f"  Pre-allocating array for {total_sequences:,} sequences...")
        all_sequences = np.zeros((total_sequences, self.window_size), dtype=np.int16)
        
        # PASS 2: Fill the pre-allocated array with sliding windows
        print(f"  Creating sliding windows (size={self.window_size}, stride={self.window_size//2})...")
        current_idx = 0
        
        for i, encoding in enumerate(encodings):
            # Convert to NumPy array for stride tricks
            token_ids = np.array(encoding.ids, dtype=np.int16)
            
            # Create sliding windows using stride tricks
            windows = self.create_sliding_windows(token_ids)
            
            if len(windows) > 0:
                # Direct assignment to pre-allocated array
                n_windows = len(windows)
                all_sequences[current_idx:current_idx + n_windows] = windows
                current_idx += n_windows
                file_sequences += n_windows
            
            file_articles += 1
            file_tokens += len(token_ids)
        
        # Save the pre-allocated array directly
        np.save(output_path, all_sequences)
        print(f"  Saved: {output_path}")
        
        # Final statistics
        elapsed = time.time() - start_time
        stats = {
            "file": file_path.name,
            "articles": file_articles,
            "tokens": file_tokens,
            "sequences": file_sequences,
            "processing_time": elapsed,
            "articles_per_sec": file_articles / elapsed if elapsed > 0 else 0
        }
        
        print(f"  Complete: {file_articles:,} articles -> {file_sequences:,} sequences")
        print(f"  Time: {elapsed:.1f}s ({stats['articles_per_sec']:.0f} articles/sec)")
        
        # Update global statistics
        self.total_articles += file_articles
        self.total_tokens += file_tokens
        self.total_sequences += file_sequences
        
        return stats
    
    def process_dataset(self, test_mode: bool = False):
        """Process all cleaned files and create tokenized dataset.
        
        Args:
            test_mode: If True, only process first file
        """
        # Setup paths
        data_dir = Path("/home/andrea/Desktop/data")
        cleaned_dir = data_dir / "cleaned_articles"
        
        if test_mode:
            output_dir = data_dir / "tokenized_datasets" / "codellama_test_dataset"
            files = [cleaned_dir / "cleaned_0.txt"]
        else:
            output_dir = data_dir / "tokenized_datasets" / "codellama_full_dataset"
            files = sorted(cleaned_dir.glob("cleaned_*.txt"))
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Dataset Tokenization Pipeline")
        print(f"{'='*60}")
        print(f"Mode: {'TEST' if test_mode else 'FULL'}")
        print(f"Window size: {self.window_size} tokens")
        print(f"Files to process: {len(files)}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
        # Process each file
        all_stats = []
        overall_start = time.time()
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing {file_path.name}")
            
            # Generate output path
            file_num = file_path.stem.split('_')[1]  # Extract number from cleaned_X.txt
            output_path = output_dir / f"tokens_{file_num}.npy"
            
            # Process file
            stats = self.process_file(file_path, output_path, test_mode=test_mode)
            all_stats.append(stats)
        
        # Generate and save metadata
        overall_elapsed = time.time() - overall_start
        metadata = {
            "mode": "test" if test_mode else "full",
            "window_size": self.window_size,
            "vocab_size": self.vocab_size,
            "total_articles": self.total_articles,
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "compression_ratio": self.total_articles / self.total_sequences if self.total_sequences > 0 else 0,
            "processing_time": overall_elapsed,
            "files_processed": len(files),
            "file_stats": all_stats
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'wb') as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"Tokenization Complete")
        print(f"{'='*60}")
        print(f"Total articles: {self.total_articles:,}")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"Total sequences: {self.total_sequences:,}")
        print(f"Average tokens/article: {self.total_tokens/self.total_articles:.0f}")
        print(f"Average sequences/article: {self.total_sequences/self.total_articles:.1f}")
        print(f"Total time: {overall_elapsed:.1f}s")
        print(f"Output saved to: {output_dir}")
        print(f"{'='*60}")


def main():
    """Main entry point for dataset preparation."""
    parser = argparse.ArgumentParser(description="Dataset Preparation Pipeline")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: process only first file")
    parser.add_argument("--window", type=int, default=512,
                       help="Context window size in tokens (default: 512)")
    
    args = parser.parse_args()
    
    # Create tokenizer instance
    dataset_tokenizer = DatasetTokenizer(
        window_size=args.window
    )
    
    # Load appropriate tokenizer
    dataset_tokenizer.load_tokenizer(test_mode=args.test)
    
    # Process dataset
    dataset_tokenizer.process_dataset(test_mode=args.test)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDataset preparation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)