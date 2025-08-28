#!/usr/bin/env python3
"""FineWeb dataset preparation pipeline for tokenizing web documents."""

import sys
import time
import argparse
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional
import numpy as np
import orjson
from numpy.lib.stride_tricks import as_strided
from tokenizer import WikipediaTokenizer
from tqdm import tqdm

# Try to import datasets library
try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Please install it with:")
    print("  pip install datasets")
    sys.exit(1)


class FineWebTokenizer:
    """Tokenizes FineWeb dataset documents for training."""
    
    def __init__(self, window_size: int = 512):
        """Initialize FineWeb tokenizer.
        
        Args:
            window_size: Context window size in tokens (default 512)
        """
        self.window_size = window_size
        self.tokenizer = None
        self.vocab_size = None
        
        # Statistics tracking
        self.total_documents = 0
        self.total_tokens = 0
        self.total_sequences = 0
        self.filtered_documents = 0
        
    def load_tokenizer(self):
        """Load CodeLlama tokenizer."""
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
    
    def create_sliding_windows(self, token_ids: np.ndarray) -> np.ndarray:
        """Create sliding window sequences using NumPy stride tricks for maximum performance.
        
        Args:
            token_ids: NumPy array of token IDs from a document
            
        Returns:
            2D NumPy array of shape (num_windows, window_size)
        """
        n_tokens = len(token_ids)
        
        # Return empty array if document is too short
        if n_tokens < self.window_size:
            return np.array([], dtype=np.int16).reshape(0, self.window_size)
        
        # Calculate stride and number of windows
        stride = self.window_size // 2
        n_windows = (n_tokens - self.window_size) // stride + 1
        
        # Use stride tricks to create windows without copying data
        windows = as_strided(
            token_ids,
            shape=(n_windows, self.window_size),
            strides=(token_ids.strides[0] * stride, token_ids.strides[0])
        )
        
        # Return contiguous copy for efficient saving
        return np.ascontiguousarray(windows, dtype=np.int16)
    
    def stream_fineweb_dataset(self, 
                              max_documents: Optional[int] = None,
                              language_score_threshold: float = 0.65,
                              min_tokens: int = 100) -> Iterator[str]:
        """Stream FineWeb dataset from HuggingFace.
        
        Args:
            max_documents: Maximum number of documents to process (None for all)
            language_score_threshold: Minimum language score for English quality
            min_tokens: Minimum tokens required in a document
            
        Yields:
            Document text strings that pass filters
        """
        print(f"\nStreaming FineWeb-10BT dataset from HuggingFace...")
        print(f"Filters: language_score > {language_score_threshold}, min_tokens > {min_tokens}")
        
        # Load dataset in streaming mode to avoid downloading all 27.6GB
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True
        )
        
        documents_yielded = 0
        documents_seen = 0
        
        # Create progress bar
        pbar = tqdm(total=max_documents, desc="Documents processed", unit="docs")
        
        for example in dataset:
            documents_seen += 1
            
            # Apply quality filters
            if example.get('language_score', 0) <= language_score_threshold:
                self.filtered_documents += 1
                continue
            
            text = example.get('text', '').strip()
            if not text:
                self.filtered_documents += 1
                continue
            
            # Quick token count estimation (rough, actual will be done during tokenization)
            estimated_tokens = len(text.split()) * 1.5  # Rough estimate
            if estimated_tokens < min_tokens:
                self.filtered_documents += 1
                continue
            
            yield text
            documents_yielded += 1
            pbar.update(1)
            
            if max_documents and documents_yielded >= max_documents:
                break
        
        pbar.close()
        
        print(f"\nStreaming complete:")
        print(f"  Documents seen: {documents_seen:,}")
        print(f"  Documents filtered: {self.filtered_documents:,}")
        print(f"  Documents yielded: {documents_yielded:,}")
    
    def process_fineweb(self, 
                       output_dir: Path,
                       test_mode: bool = False,
                       batch_size: int = 10000,
                       max_files: int = 38):  # Match Wikipedia's 38 files
        """Process FineWeb dataset and create tokenized sequences.
        
        Args:
            output_dir: Directory to save tokenized sequences
            test_mode: If True, only process 10,000 documents for testing
            batch_size: Number of documents to accumulate before saving
            max_files: Maximum number of output files to create
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"FineWeb Dataset Tokenization Pipeline")
        print(f"{'='*60}")
        print(f"Mode: {'TEST' if test_mode else 'FULL'}")
        print(f"Window size: {self.window_size} tokens")
        print(f"Batch size: {batch_size:,} documents")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
        # Determine number of documents to process
        if test_mode:
            max_documents = 10000
            print(f"Test mode: Processing first {max_documents:,} documents only")
        else:
            # For full mode, we'll process enough to fill ~38 files
            # Estimate based on Wikipedia's pattern
            max_documents = None  # Process all available
            print(f"Full mode: Processing all available documents")
        
        # Process documents in batches
        overall_start = time.time()
        current_batch = []
        current_file_idx = 0
        all_stats = []
        
        # Stream documents from FineWeb
        document_stream = self.stream_fineweb_dataset(
            max_documents=max_documents,
            language_score_threshold=0.65,
            min_tokens=100
        )
        
        print(f"\nTokenizing and creating sequences...")
        batch_start_time = time.time()
        
        for doc_idx, document in enumerate(document_stream):
            current_batch.append(document)
            
            # Process batch when it reaches batch_size
            if len(current_batch) >= batch_size:
                # Process and save current batch
                file_stats = self._process_and_save_batch(
                    current_batch, 
                    current_file_idx, 
                    output_dir,
                    batch_start_time
                )
                all_stats.append(file_stats)
                
                # Reset for next batch
                current_batch = []
                current_file_idx += 1
                batch_start_time = time.time()
                
                # Stop if we've created enough files
                if max_files and current_file_idx >= max_files:
                    print(f"\nReached maximum number of files ({max_files})")
                    break
        
        # Process remaining documents in the last batch
        if current_batch:
            file_stats = self._process_and_save_batch(
                current_batch,
                current_file_idx,
                output_dir,
                batch_start_time
            )
            all_stats.append(file_stats)
        
        # Generate and save metadata
        overall_elapsed = time.time() - overall_start
        metadata = {
            "dataset": "fineweb-10BT",
            "mode": "test" if test_mode else "full",
            "window_size": self.window_size,
            "vocab_size": self.vocab_size,
            "total_documents": self.total_documents,
            "filtered_documents": self.filtered_documents,
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "compression_ratio": self.total_documents / self.total_sequences if self.total_sequences > 0 else 0,
            "processing_time": overall_elapsed,
            "files_created": len(all_stats),
            "file_stats": all_stats
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'wb') as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"FineWeb Tokenization Complete")
        print(f"{'='*60}")
        print(f"Total documents: {self.total_documents:,}")
        print(f"Filtered documents: {self.filtered_documents:,}")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"Total sequences: {self.total_sequences:,}")
        if self.total_documents > 0:
            print(f"Average tokens/document: {self.total_tokens/self.total_documents:.0f}")
            print(f"Average sequences/document: {self.total_sequences/self.total_documents:.1f}")
        print(f"Total time: {overall_elapsed:.1f}s")
        print(f"Files created: {len(all_stats)}")
        print(f"Output saved to: {output_dir}")
        print(f"{'='*60}")
    
    def _process_and_save_batch(self, 
                               documents: List[str], 
                               file_idx: int,
                               output_dir: Path,
                               batch_start_time: float) -> Dict[str, Any]:
        """Process a batch of documents and save to file.
        
        Args:
            documents: List of document texts
            file_idx: Index for output file naming
            output_dir: Directory to save output
            batch_start_time: Time when batch processing started
            
        Returns:
            Statistics dictionary for this batch
        """
        print(f"\n  Processing batch {file_idx} ({len(documents):,} documents)...")
        
        # Get EOS token from the tokenizer
        eos_token = self.tokenizer.tokenizer.eos_token  # "</s>"
        
        # Append EOS token to each document for proper boundaries
        documents_with_eos = [doc + eos_token for doc in documents]
        
        # Batch encode all documents
        print(f"    Tokenizing {len(documents):,} documents...")
        encoded_batch = self.tokenizer.encode(documents_with_eos, add_special_tokens=True)
        
        # Convert to list of objects with 'ids' attribute for compatibility
        encodings = [type('Encoding', (), {'ids': ids})() for ids in encoded_batch]
        
        # Calculate total sequences needed for pre-allocation
        token_lengths = [len(encoding.ids) for encoding in encodings]
        total_sequences = sum(
            (length - self.window_size) // (self.window_size // 2) + 1 
            for length in token_lengths if length >= self.window_size
        )
        
        if total_sequences == 0:
            print(f"    Warning: No sequences generated from batch {file_idx}")
            return {
                "file": f"tokens_{file_idx}.npy",
                "documents": len(documents),
                "tokens": sum(token_lengths),
                "sequences": 0,
                "processing_time": time.time() - batch_start_time
            }
        
        # Pre-allocate the sequences array
        print(f"    Creating {total_sequences:,} sliding windows...")
        all_sequences = np.zeros((total_sequences, self.window_size), dtype=np.int16)
        
        # Fill the pre-allocated array with sliding windows
        current_idx = 0
        batch_documents = 0
        batch_tokens = 0
        batch_sequences = 0
        
        for encoding in encodings:
            # Convert to NumPy array
            token_ids = np.array(encoding.ids, dtype=np.int16)
            
            # Create sliding windows
            windows = self.create_sliding_windows(token_ids)
            
            if len(windows) > 0:
                n_windows = len(windows)
                all_sequences[current_idx:current_idx + n_windows] = windows
                current_idx += n_windows
                batch_sequences += n_windows
            
            batch_documents += 1
            batch_tokens += len(token_ids)
        
        # Save the sequences
        output_path = output_dir / f"tokens_{file_idx}.npy"
        np.save(output_path, all_sequences)
        print(f"    Saved: {output_path}")
        
        # Update global statistics
        self.total_documents += batch_documents
        self.total_tokens += batch_tokens
        self.total_sequences += batch_sequences
        
        # Calculate batch statistics
        elapsed = time.time() - batch_start_time
        stats = {
            "file": f"tokens_{file_idx}.npy",
            "documents": batch_documents,
            "tokens": batch_tokens,
            "sequences": batch_sequences,
            "processing_time": elapsed,
            "docs_per_sec": batch_documents / elapsed if elapsed > 0 else 0
        }
        
        print(f"    Complete: {batch_documents:,} documents -> {batch_sequences:,} sequences")
        print(f"    Time: {elapsed:.1f}s ({stats['docs_per_sec']:.0f} docs/sec)")
        
        return stats


def main():
    """Main entry point for FineWeb dataset preparation."""
    parser = argparse.ArgumentParser(description="FineWeb Dataset Preparation Pipeline")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: process only 10,000 documents")
    parser.add_argument("--window", type=int, default=512,
                       help="Context window size in tokens (default: 512)")
    parser.add_argument("--batch-size", type=int, default=10000,
                       help="Number of documents per batch (default: 10000)")
    parser.add_argument("--max-files", type=int, default=38,
                       help="Maximum number of output files (default: 38)")
    
    args = parser.parse_args()
    
    # Create tokenizer instance
    fineweb_tokenizer = FineWebTokenizer(window_size=args.window)
    
    # Load CodeLlama tokenizer
    fineweb_tokenizer.load_tokenizer()
    
    # Setup output directory
    data_dir = Path("/home/andrea/Desktop/data")
    if args.test:
        output_dir = data_dir / "tokenized_datasets" / "fineweb_test_dataset"
    else:
        output_dir = data_dir / "tokenized_datasets" / "fineweb_full_dataset"
    
    # Process FineWeb dataset
    fineweb_tokenizer.process_fineweb(
        output_dir=output_dir,
        test_mode=args.test,
        batch_size=args.batch_size,
        max_files=args.max_files
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nFineWeb preparation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)