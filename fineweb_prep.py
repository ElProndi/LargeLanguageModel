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
import threading
from queue import Queue

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
        self.stride = window_size // 2  # Pre-compute stride
        
        # Statistics tracking
        self.total_documents = 0
        self.total_tokens = 0
        self.total_sequences = 0
        self.filtered_documents = 0
        
        # Optimization: larger accumulation buffer for better tokenizer parallelization
        self.optimal_batch_size = 288000  # Sized for 10B tokens in 38 files (~915 tokens/doc average)
        self.target_tokens_per_file = 263_000_000  # 10B tokens / 38 files
        
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
    
    def calculate_num_sequences(self, token_lengths: List[int]) -> int:
        """Calculate total number of sequences that will be generated.
        
        Args:
            token_lengths: List of token counts for each document
            
        Returns:
            Total number of sequences that will be created
        """
        # Vectorized calculation for better performance
        total = sum(
            (length - self.window_size) // self.stride + 1 
            for length in token_lengths if length >= self.window_size
        )
        return total
    
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
        
        # Use pre-computed stride
        n_windows = (n_tokens - self.window_size) // self.stride + 1
        
        # Use stride tricks to create windows without copying data
        windows = as_strided(
            token_ids,
            shape=(n_windows, self.window_size),
            strides=(token_ids.strides[0] * self.stride, token_ids.strides[0])
        )
        
        # Return contiguous copy for efficient saving
        return np.ascontiguousarray(windows, dtype=np.int16)
    
    
    def stream_fineweb_dataset_with_prefetch(self,
                                            max_documents: Optional[int] = None,
                                            language_score_threshold: float = 0.65,
                                            min_tokens: int = 100,
                                            progress_update_freq: int = 1000,
                                            prefetch_size: int = 1000) -> Iterator[str]:
        """Stream FineWeb dataset with prefetching for optimal performance.
        
        Uses a separate thread to prefetch documents while processing current batch.
        
        Args:
            max_documents: Maximum number of documents to process
            language_score_threshold: Minimum language score
            min_tokens: Minimum tokens required
            progress_update_freq: Progress bar update frequency
            prefetch_size: Number of documents to prefetch
            
        Yields:
            Document text strings that pass filters
        """
        # Use the regular streaming with a prefetch buffer
        buffer = Queue(maxsize=prefetch_size)
        stop_event = threading.Event()
        
        def prefetch_worker():
            """Worker thread to prefetch documents."""
            try:
                for doc in self.stream_fineweb_dataset(
                    max_documents=max_documents,
                    language_score_threshold=language_score_threshold,
                    min_tokens=min_tokens,
                    progress_update_freq=progress_update_freq
                ):
                    if stop_event.is_set():
                        break
                    buffer.put(doc)
                buffer.put(None)  # Signal end of stream
            except Exception as e:
                buffer.put(e)  # Pass exception to main thread
        
        # Start prefetch thread
        thread = threading.Thread(target=prefetch_worker, daemon=True)
        thread.start()
        
        # Yield from buffer
        while True:
            item = buffer.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        
        stop_event.set()
        thread.join(timeout=1)
    
    def stream_fineweb_dataset(self, 
                              max_documents: Optional[int] = None,
                              language_score_threshold: float = 0.65,
                              min_tokens: int = 100,
                              progress_update_freq: int = 1000) -> Iterator[str]:
        """Stream FineWeb dataset from HuggingFace with optimized performance.
        
        Args:
            max_documents: Maximum number of documents to process (None for all)
            language_score_threshold: Minimum language score for English quality
            min_tokens: Minimum tokens required in a document
            progress_update_freq: How often to update progress bar (default every 1000 docs)
            
        Yields:
            Document text strings that pass filters
        """
        print(f"\nStreaming FineWeb-10BT dataset from HuggingFace...")
        print(f"Filters: language_score > {language_score_threshold}, min_tokens > {min_tokens}")
        print(f"Progress updates every {progress_update_freq:,} documents")
        
        # Load dataset in streaming mode to avoid downloading all 27.6GB
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True
        )
        
        documents_yielded = 0
        documents_seen = 0
        update_counter = 0
        
        # Create progress bar with less frequent updates
        pbar = tqdm(total=max_documents, desc="Documents processed", unit="docs", 
                   mininterval=0.5, smoothing=0.3)  # Reduce update frequency
        
        for example in dataset:
            documents_seen += 1
            
            # OPTIMIZATION: Check language score first (fastest filter)
            language_score = example.get('language_score', 0)
            if language_score <= language_score_threshold:
                self.filtered_documents += 1
                continue
            
            # OPTIMIZATION: Avoid strip() until necessary
            text = example.get('text', '')
            if not text:
                self.filtered_documents += 1
                continue
            
            # OPTIMIZATION: Skip expensive token estimation - just check minimum length
            # Assuming average 4.5 chars per token (rough but fast)
            if len(text) < min_tokens * 4.5:
                self.filtered_documents += 1
                continue
            
            # Only strip when yielding (lazy evaluation)
            yield text.strip() if text else text
            documents_yielded += 1
            update_counter += 1
            
            # OPTIMIZATION: Batch progress updates
            if update_counter >= progress_update_freq:
                pbar.update(update_counter)
                update_counter = 0
            
            if max_documents and documents_yielded >= max_documents:
                break
        
        # Final progress update
        if update_counter > 0:
            pbar.update(update_counter)
        pbar.close()
        
        print(f"\nStreaming complete:")
        print(f"  Documents seen: {documents_seen:,}")
        print(f"  Documents filtered: {self.filtered_documents:,}")
        print(f"  Documents yielded: {documents_yielded:,}")
    
    
    def process_fineweb(self, 
                       output_dir: Path,
                       test_mode: bool = False,
                       batch_size: Optional[int] = None,  # Auto-calculated if None
                       max_files: int = 38,  # Match Wikipedia's 38 files
                       target_total_tokens: int = 10_000_000_000):  # 10B tokens target
        """Process FineWeb dataset and create tokenized sequences.
        
        Args:
            output_dir: Directory to save tokenized sequences
            test_mode: If True, only process 10,000 documents for testing  
            batch_size: Number of documents per batch (auto-calculated if None)
            max_files: Maximum number of output files to create
            target_total_tokens: Target total tokens to process (default 10B)
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-calculate batch size for target tokens if not specified
        if batch_size is None:
            # Based on observed ~915 tokens per document average
            avg_tokens_per_doc = 915
            total_docs_needed = target_total_tokens // avg_tokens_per_doc
            batch_size = total_docs_needed // max_files
            print(f"Auto-calculated batch size: {batch_size:,} documents per file")
            print(f"(Target: {target_total_tokens/1e9:.1f}B tokens in {max_files} files)")
        
        print(f"\n{'='*60}")
        print(f"FineWeb Dataset Tokenization Pipeline")
        print(f"{'='*60}")
        print(f"Mode: {'TEST' if test_mode else 'FULL'}")
        print(f"Target: {target_total_tokens/1e9:.1f}B tokens")
        print(f"Window size: {self.window_size} tokens")
        print(f"Batch size: {batch_size:,} documents/file")
        print(f"Max files: {max_files}")
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
        
        # Stream documents from FineWeb with optimized settings
        # Use prefetching for better performance in full mode
        use_prefetch = not test_mode
        if use_prefetch:
            print("Using prefetched streaming for optimal performance...")
            document_stream = self.stream_fineweb_dataset_with_prefetch(
                max_documents=max_documents,
                language_score_threshold=0.65,
                min_tokens=100,
                progress_update_freq=1000,  # Batch progress updates
                prefetch_size=5000  # Prefetch buffer size
            )
        else:
            document_stream = self.stream_fineweb_dataset(
                max_documents=max_documents,
                language_score_threshold=0.65,
                min_tokens=100,
                progress_update_freq=1000  # Batch progress updates
            )
        
        print(f"\nTokenizing and creating sequences...")
        print(f"Using optimized batch size: {batch_size:,} documents")
        batch_start_time = time.time()
        
        for doc_idx, document in enumerate(document_stream):
            current_batch.append(document)
            
            # Process batch when it reaches batch_size
            if len(current_batch) >= batch_size:
                # Process and save current batch with optimized two-pass strategy
                file_stats = self._process_and_save_batch_optimized(
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
            file_stats = self._process_and_save_batch_optimized(
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
        
        # Batch encode all documents WITHOUT auto-added specials
        print(f"    Tokenizing {len(documents):,} documents...")
        special_ids = self.tokenizer.get_special_token_ids()
        bos_token_id = special_ids["bos_token_id"]
        eos_token_id = special_ids["eos_token_id"]
        core_encoded = self.tokenizer.encode(documents, add_special_tokens=False)
        # Manually add BOS/EOS IDs deterministically
        encoded_batch = [[bos_token_id] + ids + [eos_token_id] for ids in core_encoded]
        
        # Work directly with token IDs - no wrapper objects needed
        token_lengths = [len(ids) for ids in encoded_batch]
        total_sequences = sum(
            (length - self.window_size) // self.stride + 1 
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
        
        # Create sliding windows for all documents
        print(f"    Creating {total_sequences:,} sliding windows...")
        all_windows = []
        
        for token_ids in encoded_batch:
            # Convert to NumPy array
            token_array = np.array(token_ids, dtype=np.int16)
            
            # Create sliding windows
            windows = self.create_sliding_windows(token_array)
            
            if windows.shape[0] > 0:
                all_windows.append(windows)
        
        if all_windows:
            all_sequences = np.vstack(all_windows)
            batch_sequences = all_sequences.shape[0]
        else:
            all_sequences = np.array([], dtype=np.int16).reshape(0, self.window_size)
            batch_sequences = 0
        
        batch_documents = len(documents)
        batch_tokens = sum(token_lengths)
        
        # Save the sequences
        output_path = output_dir / f"tokens_{file_idx}.npy"
        np.save(output_path, all_sequences)
        print(f"    Saving to: {output_path}")
        
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
    
    def _process_and_save_batch_optimized(self, 
                                        documents: List[str], 
                                        file_idx: int,
                                        output_dir: Path,
                                        batch_start_time: float) -> Dict[str, Any]:
        """Process a batch of documents using optimized two-pass strategy.
        
        This implementation follows the highly optimized pattern from dataset_prep.py:
        1. Load all documents into memory
        2. Tokenize entire batch at once for maximum parallelization
        3. Calculate total sequences needed (Pass 1)
        4. Pre-allocate entire array
        5. Fill array with sliding windows (Pass 2)
        
        Args:
            documents: List of document texts
            file_idx: Index for output file naming
            output_dir: Directory to save output
            batch_start_time: Time when batch processing started
            
        Returns:
            Statistics dictionary for this batch
        """
        print(f"\n  Processing batch {file_idx} ({len(documents):,} documents) with optimized strategy...")
        
        # Get special tokens from tokenizer
        special_ids = self.tokenizer.get_special_token_ids()
        eos_token_id = special_ids["eos_token_id"]
        bos_token_id = special_ids["bos_token_id"]

        # OPTIMIZATION: Send entire batch to tokenizer at once (no specials)
        print(f"    Sending {len(documents):,} documents to tokenizer (all at once)...")
        tokenize_start = time.time()
        core_encoded = self.tokenizer.encode(documents, add_special_tokens=False)
        # Manually add BOS/EOS IDs deterministically
        encoded_batch = [[bos_token_id] + ids + [eos_token_id] for ids in core_encoded]
        tokenize_elapsed = time.time() - tokenize_start
        print(f"    Tokenization complete in {tokenize_elapsed:.1f}s ({len(documents)/tokenize_elapsed:.0f} docs/sec)")
        
        # Validation: Check special tokens in first few documents
        num_to_check = min(5, len(encoded_batch))
        eos_count = sum(1 for ids in encoded_batch[:num_to_check] if eos_token_id in ids)
        bos_count = sum(1 for ids in encoded_batch[:num_to_check] if bos_token_id in ids)
        
        # Log validation results
        if eos_count < num_to_check:
            print(f"    ⚠ WARNING: Only {eos_count}/{num_to_check} documents have EOS tokens")
        if bos_count < num_to_check:
            print(f"    ⚠ WARNING: Only {bos_count}/{num_to_check} documents have BOS tokens")
        
        # PASS 1: Calculate total sequences needed for pre-allocation
        print(f"    Pass 1: Calculating total sequences for pre-allocation...")
        token_lengths = [len(ids) for ids in encoded_batch]
        total_sequences = self.calculate_num_sequences(token_lengths)
        
        if total_sequences == 0:
            print(f"    Warning: No sequences generated from batch {file_idx}")
            return {
                "file": f"tokens_{file_idx}.npy",
                "documents": len(documents),
                "tokens": sum(token_lengths),
                "sequences": 0,
                "processing_time": time.time() - batch_start_time
            }
        
        # OPTIMIZATION: Pre-allocate the entire sequences array
        print(f"    Pre-allocating array for {total_sequences:,} sequences...")
        all_sequences = np.zeros((total_sequences, self.window_size), dtype=np.int16)
        
        # PASS 2: Fill the pre-allocated array with sliding windows
        print(f"    Pass 2: Creating sliding windows (size={self.window_size}, stride={self.stride})...")
        current_idx = 0
        windows_start = time.time()
        
        # Process each document's tokens
        batch_documents = 0
        batch_tokens = 0
        
        for token_ids in encoded_batch:
            # Convert to NumPy array once
            token_array = np.array(token_ids, dtype=np.int16)
            
            # Create sliding windows using stride tricks
            windows = self.create_sliding_windows(token_array)
            
            if len(windows) > 0:
                # Direct assignment to pre-allocated array (no vstack needed!)
                n_windows = len(windows)
                all_sequences[current_idx:current_idx + n_windows] = windows
                current_idx += n_windows
            
            batch_documents += 1
            batch_tokens += len(token_ids)
        
        windows_elapsed = time.time() - windows_start
        print(f"    Window creation complete in {windows_elapsed:.1f}s ({total_sequences/windows_elapsed:.0f} windows/sec)")
        
        # Save the pre-allocated array directly
        output_path = output_dir / f"tokens_{file_idx}.npy"
        print(f"    Saving to: {output_path}")
        save_start = time.time()
        np.save(output_path, all_sequences)
        save_elapsed = time.time() - save_start
        print(f"    Save complete in {save_elapsed:.1f}s")
        
        # Update global statistics
        self.total_documents += batch_documents
        self.total_tokens += batch_tokens
        self.total_sequences += total_sequences
        
        # Calculate batch statistics
        elapsed = time.time() - batch_start_time
        stats = {
            "file": f"tokens_{file_idx}.npy",
            "documents": batch_documents,
            "tokens": batch_tokens,
            "sequences": total_sequences,
            "processing_time": elapsed,
            "docs_per_sec": batch_documents / elapsed if elapsed > 0 else 0,
            "tokens_per_sec": batch_tokens / elapsed if elapsed > 0 else 0,
            "breakdown": {
                "tokenization_time": tokenize_elapsed,
                "window_creation_time": windows_elapsed,
                "save_time": save_elapsed
            }
        }
        
        print(f"    Complete: {batch_documents:,} documents -> {total_sequences:,} sequences")
        print(f"    Performance: {stats['docs_per_sec']:.0f} docs/sec, {stats['tokens_per_sec']:.0f} tokens/sec")
        print(f"    Total time: {elapsed:.1f}s")
        
        return stats


def main():
    """Main entry point for FineWeb dataset preparation."""
    parser = argparse.ArgumentParser(description="FineWeb Dataset Preparation Pipeline")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: process only 10,000 documents")
    parser.add_argument("--window", type=int, default=512,
                       help="Context window size in tokens (default: 512)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Number of documents per batch (default: auto-calculated for 10B tokens)")
    parser.add_argument("--target-tokens", type=int, default=10_000_000_000,
                       help="Target total tokens to process (default: 10B)")
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
        max_files=args.max_files,
        target_total_tokens=args.target_tokens
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
