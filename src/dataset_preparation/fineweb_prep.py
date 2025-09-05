#!/usr/bin/env python3
"""FineWeb dataset preparation pipeline - processes downloaded JSONL files

This module reads pre-downloaded FineWeb data from JSONL files and tokenizes it
for training. It expects data to be downloaded first using fineweb_download.py.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from itertools import chain
import numpy as np
import json
from tqdm import tqdm
from .tokenizer import CodeLlamaTokenizer
from collections import deque
import gc  # For explicit garbage collection


class FineWebProcessor:
    """Processes pre-downloaded FineWeb JSONL files into tokenized sequences."""
    
    def __init__(self, window_size: int = 2048):
        """Initialize processor with window size."""
        self.window_size = window_size
        self.tokenizer = None
        self.vocab_size = None
        
        # Continuous streaming buffer - using numpy array for efficiency
        self.token_buffer = np.array([], dtype=np.int16)  # Persists across files
        
        # Statistics tracking
        self.total_documents = 0
        self.total_tokens = 0
        self.total_sequences = 0
        
        # Optional preallocations (filled in process_fineweb when batch_size known)
        self._ends_buf: Optional[np.ndarray] = None  # int64 cumsum workspace
        self.expected_batch_size: Optional[int] = None
    
    def load_tokenizer(self):
        """Load the CodeLlama tokenizer."""
        tokenizer_path = Path("tokenizers/codellama_tokenizer")
        
        print(f"Loading tokenizer...")
        self.tokenizer = CodeLlamaTokenizer()
        
        # Try to load saved tokenizer first, otherwise download
        if tokenizer_path.exists():
            print(f"  Loading from {tokenizer_path}")
            self.tokenizer.load(str(tokenizer_path))
        else:
            print(f"  Downloading CodeLlama tokenizer from HuggingFace...")
            self.tokenizer.train()  # Downloads the pre-trained tokenizer
            self.tokenizer.save(str(tokenizer_path))
        
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"  Vocabulary size: {self.vocab_size:,}")
    
    def pack_documents_continuous(self, tokenized_docs: List[np.ndarray]) -> np.ndarray:
        """Pack documents into fixed-size sequences without gaps or padding.
        
        Optimized version using numpy operations throughout.
        
        Args:
            tokenized_docs: List of tokenized documents (with BOS/EOS already added)
            
        Returns:
            Numpy array of completed sequences (shape: [n_sequences, window_size])
        """
        # Fast path: single contiguous 1-D array of tokens
        if len(tokenized_docs) == 1 and isinstance(tokenized_docs[0], np.ndarray) and tokenized_docs[0].ndim == 1:
            new = tokenized_docs[0]
            buf_len = len(self.token_buffer)
            new_len = len(new)
            total_tokens_available = buf_len + new_len
            n_sequences = total_tokens_available // self.window_size

            if n_sequences == 0:
                # Not enough to form a full window: append to buffer
                if new_len:
                    if buf_len:
                        self.token_buffer = np.concatenate([self.token_buffer, new])
                    else:
                        # Reuse reference if buffer empty to avoid an extra copy
                        self.token_buffer = new.copy()
                return np.array([], dtype=np.int16).reshape(0, self.window_size)

            sequences = np.empty((n_sequences, self.window_size), dtype=np.int16)

            # Fill first sequence with buffer + prefix of new
            row = 0
            consumed = 0
            if buf_len:
                need = self.window_size - buf_len
                sequences[0, :buf_len] = self.token_buffer
                sequences[0, buf_len:buf_len + need] = new[:need]
                consumed = need
                row = 1

            # Fill remaining full sequences directly from new
            remaining = new[consumed:]
            if remaining.size:
                full = remaining.size // self.window_size
                if full:
                    sequences[row:row + full] = remaining[:full * self.window_size].reshape(full, self.window_size)
                    row += full
                # Set new buffer to leftover
                self.token_buffer = remaining[full * self.window_size:]
            else:
                self.token_buffer = np.array([], dtype=np.int16)

            # Reset buffer if we consumed exactly into windows
            if not buf_len:
                # Nothing to clear beyond what we set above
                pass
            else:
                # Buffer always fully consumed here
                # Already overwritten by the new leftover above
                pass

            return sequences

        # Generic path: fall back to concatenation
        total_new_tokens = sum(len(doc) for doc in tokenized_docs)
        total_tokens_available = len(self.token_buffer) + total_new_tokens
        n_sequences = total_tokens_available // self.window_size
        if n_sequences == 0:
            if len(tokenized_docs) > 0:
                parts = [self.token_buffer] if len(self.token_buffer) > 0 else []
                parts.extend(tokenized_docs)
                self.token_buffer = np.concatenate(parts)
            return np.array([], dtype=np.int16).reshape(0, self.window_size)

        parts = [self.token_buffer] if len(self.token_buffer) > 0 else []
        parts.extend(tokenized_docs)
        concatenated = np.concatenate(parts)
        sequences = np.empty((n_sequences, self.window_size), dtype=np.int16)
        total_tokens_to_use = n_sequences * self.window_size
        sequences[:] = concatenated[:total_tokens_to_use].reshape(n_sequences, self.window_size)
        self.token_buffer = concatenated[total_tokens_to_use:]
        return sequences
    
    def read_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read documents from a JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        # Time file reading and JSON parsing separately
        read_start = time.time()
        lines_read = 0
        json_parse_time = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                lines_read += 1
                json_start = time.time()
                doc = json.loads(line)
                json_parse_time += time.time() - json_start
                documents.append(doc)
        
        total_read_time = time.time() - read_start
        file_io_time = total_read_time - json_parse_time
        
        print(f"    JSONL Reading Timing:")
        print(f"      - Total time: {total_read_time:.2f}s")
        print(f"      - File I/O: {file_io_time:.2f}s")
        print(f"      - JSON parsing: {json_parse_time:.2f}s")
        print(f"      - Lines/documents: {lines_read:,}")
        print(f"      - Docs/second: {lines_read/total_read_time:.1f}")
        
        return documents
    
    def process_documents(self, documents: List[str], output_dir: Path, file_idx: int, is_final: bool = False) -> Dict[str, Any]:
        """Process a list of documents: tokenize, create windows, and save.
        
        Args:
            documents: List of document texts
            output_dir: Directory to save output
            file_idx: Index for output file naming
            is_final: Whether this is the final batch
            
        Returns:
            Statistics for this batch
        """
        batch_start = time.time()
        
        # Get special tokens
        special_ids = self.tokenizer.get_special_token_ids()
        eos_token_id = special_ids["eos_token_id"]
        bos_token_id = special_ids["bos_token_id"]
        
        # Tokenization
        tokenization_start = time.time()
        
        # Tokenize all documents in batch (without special tokens)
        encode_start = time.time()
        tokenized_docs = self.tokenizer.encode(documents, add_special_tokens=False)
        encode_time = time.time() - encode_start
        
        # Add BOS/EOS tokens to each document - Fully vectorized and allocation-light
        special_tokens_start = time.time()

        if len(tokenized_docs) > 0:
            # Calculate document lengths with BOS/EOS
            # Use numpy once and avoid per-token index creation
            doc_core_lengths = np.fromiter((len(t) for t in tokenized_docs), count=len(tokenized_docs), dtype=np.int32)
            doc_lengths = doc_core_lengths + 2  # +2 for BOS/EOS per doc

            # Pre-allocate single large array for all documents
            total_size = int(doc_lengths.sum())
            all_tokens = np.empty(total_size, dtype=np.int16)

            # Cumulative ends; derive BOS/EOS without materializing doc_positions
            n_docs = len(doc_lengths)
            # Use preallocated ends buffer when available to avoid reallocations
            if self._ends_buf is not None and self._ends_buf.shape[0] >= n_docs:
                ends = self._ends_buf[:n_docs]
                np.cumsum(doc_lengths, dtype=np.int64, out=ends)
            else:
                ends = doc_lengths.cumsum(dtype=np.int64)
            bos_positions = ends - doc_lengths
            eos_positions = ends - 1

            # Create a boolean mask for payload (non-special) positions
            payload_mask = np.ones(total_size, dtype=bool)
            payload_mask[bos_positions] = False
            payload_mask[eos_positions] = False

            # Assign BOS/EOS in one shot
            all_tokens[bos_positions] = bos_token_id
            all_tokens[eos_positions] = eos_token_id

            # Concatenate all source tokens once and place via mask
            # Use fromiter over a single chained iterator to avoid per-doc array allocations
            concatenated_payload = np.fromiter(
                chain.from_iterable(tokenized_docs),
                dtype=np.int16,
                count=int((doc_lengths - 2).sum())
            )
            all_tokens[payload_mask] = concatenated_payload

            # Keep as contiguous array (single block); packer accepts list of arrays
            processed_docs = [all_tokens]
        else:
            processed_docs = []
        
        special_tokens_time = time.time() - special_tokens_start
        
        tokenization_time = time.time() - tokenization_start
        
        # Continuous packing
        packing_start = time.time()
        
        # Track statistics (use precomputed lengths when available)
        batch_documents = len(tokenized_docs)
        try:
            batch_tokens = int(doc_lengths.sum())
        except Exception:
            # Fallback (empty batch path)
            batch_tokens = 0
        
        # Pack documents into continuous sequences (now returns numpy array directly)
        all_sequences = self.pack_documents_continuous(processed_docs)
        
        # Handle final batch - save any remaining tokens in buffer
        if is_final and len(self.token_buffer) > 0:
            # Pad the final sequence with EOS tokens
            final_sequence = np.full((1, self.window_size), eos_token_id, dtype=np.int16)
            final_sequence[0, :len(self.token_buffer)] = self.token_buffer
            # Concatenate with existing sequences
            if all_sequences.shape[0] > 0:
                all_sequences = np.vstack([all_sequences, final_sequence])
            else:
                all_sequences = final_sequence
            self.token_buffer = np.array([], dtype=np.int16)  # Clear buffer
        
        # No need for separate numpy conversion timing since we already have numpy array
        numpy_time = 0.0  # For backward compatibility with timing reports
        
        packing_time = time.time() - packing_start
        
        # File I/O
        output_path = output_dir / f"tokens_{file_idx}.npy"
        io_start = time.time()
        np.save(output_path, all_sequences)
        io_time = time.time() - io_start
        
        # Calculate total processing time
        processing_time = time.time() - batch_start
        
        # Update global statistics
        self.total_documents += batch_documents
        self.total_tokens += batch_tokens
        self.total_sequences += len(all_sequences)
        
        # Print detailed timing breakdown
        print(f"    Batch {file_idx} timing breakdown:")
        print(f"      - Total processing: {processing_time:.2f}s")
        print(f"      - Tokenization: {tokenization_time:.2f}s ({tokenization_time/processing_time*100:.1f}%)")
        print(f"        - Encoding: {encode_time:.2f}s")
        print(f"        - Special tokens: {special_tokens_time:.2f}s")
        print(f"      - Packing: {packing_time:.2f}s ({packing_time/processing_time*100:.1f}%)")
        print(f"        - Numpy vstack: {numpy_time:.2f}s")
        print(f"      - File I/O: {io_time:.2f}s ({io_time/processing_time*100:.1f}%)")
        print(f"      - Throughput: {batch_tokens/processing_time:.0f} tokens/s")
        
        # Prepare return statistics
        stats = {
            "file": f"tokens_{file_idx}.npy",
            "documents": batch_documents,
            "tokens": batch_tokens,
            "sequences": len(all_sequences),
            "processing_time": processing_time,
            "tokenization_time": tokenization_time,
            "packing_time": packing_time,
            "io_time": io_time,
            "encode_time": encode_time,
            "special_tokens_time": special_tokens_time,
            "numpy_time": numpy_time
        }
        
        # Memory cleanup after we're done with the data
        del all_sequences
        del processed_docs
        del tokenized_docs
        gc.collect()  # Force garbage collection
        
        return stats
    
    def process_fineweb(self,
                        input_dir: Path,
                        output_dir: Path,
                        test_mode: bool = False,
                        batch_size: int = 700000):
        """Main processing pipeline that reads JSONL files from disk.

        Args:
            input_dir: Directory containing downloaded JSONL files
            output_dir: Directory to save tokenized sequences
            test_mode: If True, process only one chunk file for testing
            batch_size: Number of documents per output file
        """
        # Validate input directory
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            print("Please run fineweb_download.py first to download the dataset")
            sys.exit(1)
        
        # Check for metadata file
        metadata_file = input_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"Error: No metadata.json found in {input_dir}")
            print("The download may be incomplete. Please run fineweb_download.py")
            sys.exit(1)
        
        # Record expected batch size for preallocations
        self.expected_batch_size = int(batch_size)
        self._ends_buf = np.empty(self.expected_batch_size, dtype=np.int64)

        # Load download metadata
        with open(metadata_file, 'r') as f:
            download_metadata = json.load(f)
        
        if not download_metadata.get('complete', False):
            print(f"Warning: Download is marked as incomplete")
            print(f"  Documents downloaded: {download_metadata.get('total_documents', 0):,}")
            response = input("\nContinue processing anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        
        # Find all chunk files
        chunk_files = sorted(input_dir.glob("fineweb_chunk_*.jsonl"))
        if not chunk_files:
            print(f"Error: No chunk files found in {input_dir}")
            sys.exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine files to process
        if test_mode:
            chunk_files = chunk_files[:1]  # Process only first chunk
            print(f"\n{'='*60}")
            print(f"FineWeb Dataset Tokenization (TEST MODE)")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"FineWeb Dataset Tokenization (FULL MODE)")
            print(f"{'='*60}")
        
        print(f"Configuration:")
        print(f"  Input directory: {input_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Window size: {self.window_size} tokens")
        print(f"  Documents per output file: {batch_size:,}")
        print(f"  Chunk files to process: {len(chunk_files)}")
        print(f"{'='*60}\n")
        
        all_stats: List[Dict[str, Any]] = []
        overall_start = time.time()
        output_file_idx = 0
        documents_buffer = []
        
        # Process each chunk file
        for chunk_idx, chunk_file in enumerate(chunk_files):
            # Read documents from JSONL file
            chunk_start = time.time()
            documents_in_chunk = self.read_jsonl_file(chunk_file)
            read_time = time.time() - chunk_start
            
            print(f"\n[Chunk {chunk_idx}] Loaded {len(documents_in_chunk):,} documents from {chunk_file.name} in {read_time:.1f}s")
            
            # Extract text from documents
            extract_start = time.time()
            docs_extracted = 0
            for doc in documents_in_chunk:
                text = doc.get('text', '')
                if text:
                    documents_buffer.append(text)
                    docs_extracted += 1
                    
                    # Process batch when buffer is full
                    if len(documents_buffer) >= batch_size:
                        batch_docs = documents_buffer[:batch_size]
                        documents_buffer = documents_buffer[batch_size:]
                        
                        print(f"  Processing output file {output_file_idx}: {len(batch_docs):,} documents")
                        stats = self.process_documents(
                            batch_docs, 
                            output_dir, 
                            output_file_idx,
                            is_final=False
                        )
                        all_stats.append(stats)
                        output_file_idx += 1
                        
                        # Memory cleanup after processing batch
                        del batch_docs
                        gc.collect()  # Force garbage collection to free memory immediately
            
            extract_time = time.time() - extract_start
            print(f"    Text extraction: {extract_time:.2f}s for {docs_extracted:,} documents ({docs_extracted/extract_time:.1f} docs/s)")
            
            # Memory cleanup after processing each chunk file
            del documents_in_chunk
            gc.collect()  # Force garbage collection after chunk processing
        
        # Process any remaining documents
        if documents_buffer:
            print(f"\n[Final] Processing remaining {len(documents_buffer):,} documents")
            stats = self.process_documents(
                documents_buffer,
                output_dir,
                output_file_idx,
                is_final=True  # This is the final batch, so handle buffer
            )
            all_stats.append(stats)
            output_file_idx += 1
            
            # Clear the documents buffer after final processing
            del documents_buffer
            gc.collect()
        elif len(self.token_buffer) > 0:
            # Save any remaining tokens in buffer as a final file
            print(f"\n[Final] Saving remaining {len(self.token_buffer):,} tokens")
            # Get EOS token ID
            special_ids = self.tokenizer.get_special_token_ids()
            eos_token_id = special_ids["eos_token_id"]
            
            # Pad final sequence
            final_sequence = np.full(self.window_size, eos_token_id, dtype=np.int16)
            final_sequence[:len(self.token_buffer)] = self.token_buffer
            
            # Save
            output_path = output_dir / f"tokens_{output_file_idx}.npy"
            np.save(output_path, final_sequence.reshape(1, -1))
            
            self.total_sequences += 1
            all_stats.append({
                "file": f"tokens_{output_file_idx}.npy",
                "documents": 0,
                "tokens": len(self.token_buffer),
                "sequences": 1,
                "processing_time": 0
            })
        
        # Calculate total time
        overall_time = time.time() - overall_start
        
        # Save metadata
        metadata = {
            "dataset": "fineweb-100BT",
            "mode": "test" if test_mode else "full",
            "window_size": self.window_size,
            "vocab_size": self.vocab_size,
            "total_documents": self.total_documents,
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "processing_time": overall_time,
            "files_created": len(all_stats),
            "chunks_processed": len(chunk_files),
            "file_stats": all_stats
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Calculate timing aggregates
        total_encode_time = sum(stat.get('encode_time', 0) for stat in all_stats)
        total_special_tokens_time = sum(stat.get('special_tokens_time', 0) for stat in all_stats)
        total_numpy_time = sum(stat.get('numpy_time', 0) for stat in all_stats)
        total_tokenization = sum(stat['tokenization_time'] for stat in all_stats)
        total_packing = sum(stat['packing_time'] for stat in all_stats)
        total_io = sum(stat['io_time'] for stat in all_stats)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"Tokenization Complete")
        print(f"{'='*60}")
        print(f"Statistics:")
        print(f"  Total documents: {self.total_documents:,}")
        print(f"  Total tokens: {self.total_tokens:,}")
        print(f"  Total sequences: {self.total_sequences:,}")
        print(f"  Chunks processed: {len(chunk_files)}")
        print(f"  Files created: {len(all_stats)}")
        print(f"  Total time: {overall_time:.1f}s")
        
        print(f"\nTiming Breakdown:")
        print(f"  Total processing time: {overall_time:.2f}s")
        print(f"  - Tokenization: {total_tokenization:.2f}s ({total_tokenization/overall_time*100:.1f}%)")
        print(f"    - Encoding: {total_encode_time:.2f}s")
        print(f"    - Special tokens: {total_special_tokens_time:.2f}s")
        print(f"  - Packing: {total_packing:.2f}s ({total_packing/overall_time*100:.1f}%)")
        print(f"    - Numpy operations: {total_numpy_time:.2f}s")
        print(f"  - File I/O: {total_io:.2f}s ({total_io/overall_time*100:.1f}%)")
        print(f"  - Other overhead: {overall_time - total_tokenization - total_packing - total_io:.2f}s")
        
        if self.total_documents > 0:
            print(f"\nPerformance Metrics:")
            print(f"  Tokens/document: {self.total_tokens/self.total_documents:.0f}")
            print(f"  Sequences/document: {self.total_sequences/self.total_documents:.1f}")
            print(f"  Documents/second: {self.total_documents/overall_time:.1f}")
            print(f"  Tokens/second: {self.total_tokens/overall_time:.0f}")
            print(f"  MB/second: {(self.total_tokens * 2) / (1024*1024) / overall_time:.1f} (assuming int16)")
        
        print(f"\nOutput saved to: {output_dir}")
        print(f"{'='*60}")


def main():
    """Main entry point for the FineWeb processor."""
    parser = argparse.ArgumentParser(description="FineWeb Dataset Processing - Tokenize downloaded JSONL files")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: process only one chunk file")
    parser.add_argument("--input", type=str, default=None,
                       help="Input directory containing JSONL files (default: data/raw/fineweb)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for tokenized files (default: auto-generated based on mode)")
    parser.add_argument("--window", type=int, default=2048,
                       help="Context window size in tokens (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=200000,
                       help="Number of documents per output file (default: 700000, exactly 7 chunks)")
    
    args = parser.parse_args()
    
    # Set input directory
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = Path("data/raw/fineweb")
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        data_dir = Path("data")
        if args.test:
            output_dir = data_dir / "tokenized_datasets" / "fineweb_test_dataset"
        else:
            output_dir = data_dir / "tokenized_datasets" / "fineweb_full_dataset"
    
    # Create processor instance
    processor = FineWebProcessor(window_size=args.window)
    
    # Load CodeLlama tokenizer
    processor.load_tokenizer()
    
    # Process FineWeb dataset
    try:
        processor.process_fineweb(
            input_dir=input_dir,
            output_dir=output_dir,
            test_mode=args.test,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
