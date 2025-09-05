#!/usr/bin/env python3
"""FineWeb dataset optimized downloader with efficient sharding.

This module downloads the FineWeb-100BT dataset using multiple worker processes
with dataset sharding for truly parallel downloading - each worker downloads only
its portion of data, eliminating redundant downloads.

Architecture:
- Main process: Coordination and progress tracking
- N worker processes: Parallel downloading via dataset sharding
- 1 writer process: Sequential disk writes (not a bottleneck)

Features:
- Efficient sharding: Each worker downloads ONLY its data (no redundancy)
- 4-8x bandwidth savings compared to skip/take approach
- Zero inter-worker communication needed
- Minimal IPC overhead (only data to writer)
- Memory-efficient queue with backpressure
"""

import sys
import json
import time
import signal
import argparse
import random
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from queue import Empty
from tqdm import tqdm

# Try to import datasets library
try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Please install it with:")
    print("  pip install datasets")
    sys.exit(1)


@dataclass
class WorkerConfig:
    """Configuration for a download worker."""
    worker_id: int
    total_workers: int
    chunk_size: int
    max_documents: Optional[int]  # Total documents to process (optional limit)


class DownloadWorker:
    """Worker process that downloads its shard of the dataset efficiently."""
    
    def __init__(self, config: WorkerConfig, output_queue: mp.Queue, progress_queue: mp.Queue):
        """Initialize download worker.
        
        Args:
            config: Worker configuration with sharding info
            output_queue: Queue for sending chunks to writer
            progress_queue: Queue for progress updates to main
        """
        self.config = config
        self.output_queue = output_queue
        self.progress_queue = progress_queue
        
        # Worker-local statistics
        self.documents_processed = 0
        self.bytes_downloaded = 0
        
    def run(self):
        """Main worker loop - downloads only this worker's shard of data."""
        try:
            # Load dataset in streaming mode
            dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-100BT",
                split="train",
                streaming=True
            )
            
            # Use efficient sharding - each worker only downloads its portion!
            # This avoids the massive redundancy of skip/take approach
            worker_dataset = dataset.shard(
                num_shards=self.config.total_workers,
                index=self.config.worker_id
            )
            
            # Apply document limit if specified (for test mode)
            if self.config.max_documents:
                docs_per_worker = self.config.max_documents // self.config.total_workers
                # Give extra docs to first workers if not evenly divisible
                if self.config.worker_id < (self.config.max_documents % self.config.total_workers):
                    docs_per_worker += 1
                worker_dataset = worker_dataset.take(docs_per_worker)
            
            # Process documents in chunks
            chunk_docs = []
            docs_in_chunk = 0
            chunk_id = self.config.worker_id * 1000  # Unique chunk IDs per worker
            
            # Process documents
            for doc_idx, example in enumerate(worker_dataset):
                try:
                    # Extract text
                    text = example.get('text', '')
                    if not text:
                        continue  # Skip empty docs
                    
                    # Add to current chunk
                    # Use worker_id and doc_idx to create unique document IDs
                    chunk_docs.append({
                        'id': f"w{self.config.worker_id}_d{doc_idx}",
                        'text': text,
                        'length': len(text)
                    })
                    docs_in_chunk += 1
                    self.documents_processed += 1
                    self.bytes_downloaded += len(text)
                    
                    # Send chunk when full
                    if docs_in_chunk >= self.config.chunk_size:
                        self.send_chunk(chunk_id, chunk_docs)
                        
                        # Reset for next chunk
                        chunk_docs = []
                        docs_in_chunk = 0
                        chunk_id += 1
                        
                    # Send progress update every 1000 docs
                    if self.documents_processed % 1000 == 0:
                        self.send_progress()
                    
                except Exception as e:
                    # Log error and skip the document
                    self.progress_queue.put({
                        'worker_id': self.config.worker_id,
                        'status': f'Skipping doc {doc_idx} due to error: {e}',
                        'documents': self.documents_processed,
                        'bytes': self.bytes_downloaded
                    })
                    continue
            
            # Send remaining documents
            if chunk_docs:
                self.send_chunk(chunk_id, chunk_docs)
            
            # Send final progress
            self.send_progress()
            
            # Signal worker completion
            self.output_queue.put(('WORKER_DONE', self.config.worker_id, None))
            
        except Exception as e:
            # Send error signal
            self.output_queue.put(('ERROR', self.config.worker_id, str(e)))
        
    def send_chunk(self, chunk_id: int, documents: List[Dict]):
        """Send a chunk of documents to the writer.
        
        Args:
            chunk_id: Unique identifier for this chunk
            documents: List of document dictionaries
        """
        self.output_queue.put(('CHUNK', chunk_id, documents))
        
    def send_progress(self):
        """Send progress update to main process."""
        self.progress_queue.put({
            'worker_id': self.config.worker_id,
            'documents': self.documents_processed,
            'bytes': self.bytes_downloaded
        })


class WriterProcess:
    """Single writer process that handles all disk I/O."""
    
    def __init__(self, input_queue: mp.Queue, output_dir: Path, chunk_size: int):
        """Initialize writer process.
        
        Args:
            input_queue: Queue to receive chunks from workers
            output_dir: Directory to save JSONL files
            chunk_size: Documents per file (for naming)
        """
        self.input_queue = input_queue
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Writing state
        self.chunk_counter = 0
        self.workers_completed = 0
        self.total_documents = 0
        self.total_bytes = 0
        
    def run(self, total_workers: int):
        """Main writer loop - writes chunks to disk.
        
        Args:
            total_workers: Number of workers to wait for
        """
        while self.workers_completed < total_workers:
            try:
                # Get next item from queue (timeout prevents hanging)
                msg_type, chunk_id, data = self.input_queue.get(timeout=1.0)
                
                if msg_type == 'CHUNK':
                    # Write chunk to disk
                    self.write_chunk(chunk_id, data)
                    
                elif msg_type == 'WORKER_DONE':
                    self.workers_completed += 1
                    
                elif msg_type == 'ERROR':
                    print(f"\nWorker {chunk_id} error: {data}")
                    self.workers_completed += 1
                    
            except Empty:
                # Queue is empty, continue waiting
                continue
            except Exception as e:
                print(f"\nWriter error: {e}")
                break
        
        # Save final metadata
        self.save_metadata()
        
    def write_chunk(self, chunk_id: int, documents: List[Dict]):
        """Write a chunk of documents to disk.
        
        Args:
            chunk_id: Chunk identifier
            documents: List of document dictionaries
        """
        # Generate filename
        chunk_filename = f"fineweb_chunk_{self.chunk_counter:05d}.jsonl"
        chunk_path = self.output_dir / chunk_filename
        
        # Write documents
        with open(chunk_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                json.dump(doc, f, ensure_ascii=False)
                f.write('\n')
                self.total_documents += 1
                self.total_bytes += doc['length']
        
        self.chunk_counter += 1
        
    def save_metadata(self):
        """Save metadata about the download."""
        metadata = {
            'total_documents': self.total_documents,
            'total_bytes': self.total_bytes,
            'chunk_files': self.chunk_counter,
            'complete': self.workers_completed > 0
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


class FineWebFastDownloader:
    """Multiprocessing coordinator for fast FineWeb downloads."""
    
    def __init__(self,
                 output_dir: Path,
                 num_workers: int = 8,
                 chunk_size: int = 100000,
                 queue_size: int = 100):
        """Initialize the fast downloader.
        
        Args:
            output_dir: Directory to save JSONL files
            num_workers: Number of parallel download workers
            chunk_size: Documents per chunk file
            queue_size: Maximum chunks in queue
        """
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.queue_size = queue_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    
    def download(self, max_documents: Optional[int] = None, test_mode: bool = False):
        """Download FineWeb dataset using multiple workers.
        
        Args:
            max_documents: Maximum documents to download (None for all)
            test_mode: If True, download only a small sample
        """
        start_time = time.time()
        
        # Determine total documents
        if test_mode:
            total_documents = self.chunk_size * 2  # Two chunks for testing
        elif max_documents:
            total_documents = max_documents
        else:
            # FineWeb-100BT contains ~148 million documents (100 billion tokens)
            total_documents = 148_000_000
            print(f"Downloading complete FineWeb-100BT dataset (~148M documents, 100B tokens, ~277GB)")
        
        print(f"\nFast Multiprocessing Downloader (Optimized with Sharding)")
        print(f"=" * 60)
        print(f"Workers: {self.num_workers}")
        print(f"Documents to download: {total_documents:,}")
        print(f"Documents per worker: ~{total_documents//self.num_workers:,}")
        print(f"Chunk size: {self.chunk_size:,} docs/file")
        print(f"Output directory: {self.output_dir}")
        print(f"Using efficient sharding - each worker downloads only its portion!")
        print("=" * 60)
        
        # Create queues
        output_queue = mp.Queue(maxsize=self.queue_size)
        progress_queue = mp.Queue()
        
        # Start writer process
        writer = WriterProcess(output_queue, self.output_dir, self.chunk_size)
        writer_process = mp.Process(target=writer.run, args=(self.num_workers,))
        writer_process.start()
        
        # Start worker processes with staggered startup to avoid rate limiting
        workers = []
        startup_delay = 3.0  # Delay between worker startups (seconds)
        
        print(f"\nStarting {self.num_workers} workers with {startup_delay}s staggered startup...")
        
        for i in range(self.num_workers):
            config = WorkerConfig(
                worker_id=i,
                total_workers=self.num_workers,
                chunk_size=self.chunk_size,
                max_documents=total_documents if (test_mode or max_documents) else None
            )
            
            worker = DownloadWorker(config, output_queue, progress_queue)
            process = mp.Process(target=worker.run)
            process.start()
            workers.append(process)
            
            docs_estimate = total_documents // self.num_workers
            print(f"Started worker {i}: processing ~{docs_estimate:,} documents (shard {i}/{self.num_workers})")
            
            # Stagger worker startup to prevent simultaneous API hits
            if i < self.num_workers - 1:  # Don't delay after the last worker
                print(f"  Waiting {startup_delay}s before starting next worker...")
                time.sleep(startup_delay)
        
        # Monitor progress
        print("\nDownloading...")
        pbar = tqdm(total=total_documents, unit='docs', smoothing=0.1)
        
        total_docs_downloaded = 0
        total_bytes_downloaded = 0
        worker_status = {}
        
        # Monitor until all workers complete
        workers_alive = len(workers)
        while workers_alive > 0:
            # Check progress updates
            while not progress_queue.empty():
                try:
                    update = progress_queue.get_nowait()
                    worker_id = update['worker_id']
                    
                    # Check if this is a status message (rate limiting, etc.)
                    if 'status' in update:
                        # Show status message in progress bar description
                        pbar.set_description(f"Worker {worker_id}: {update['status']}")
                    
                    # Update total counts
                    if worker_id in worker_status:
                        old_docs = worker_status[worker_id].get('documents', 0)
                        new_docs = update.get('documents', 0)
                        if new_docs > old_docs:
                            pbar.update(new_docs - old_docs)
                            total_docs_downloaded += new_docs - old_docs
                        
                        # Recalculate total bytes
                        worker_status[worker_id] = update
                        total_bytes_downloaded = sum(
                            w.get('bytes', 0) for w in worker_status.values()
                        )
                    else:
                        docs = update.get('documents', 0)
                        if docs > 0:
                            pbar.update(docs)
                            total_docs_downloaded += docs
                            total_bytes_downloaded += update.get('bytes', 0)
                        worker_status[worker_id] = update
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'workers': workers_alive,
                        'GB': f"{total_bytes_downloaded/1e9:.2f}"
                    })
                    
                except Empty:
                    break
            
            # Check if workers are still alive
            workers_alive = sum(1 for w in workers if w.is_alive())
            time.sleep(0.1)
        
        pbar.close()
        
        # Wait for all processes to complete
        for worker in workers:
            worker.join()
        
        writer_process.join()
        
        # Print summary
        download_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Total documents: {total_docs_downloaded:,}")
        print(f"Total size: {total_bytes_downloaded / 1e9:.2f} GB")
        print(f"Download time: {download_time/60:.1f} minutes")
        print(f"Average speed: {total_docs_downloaded/download_time:.1f} docs/sec")
        print(f"Throughput: {total_bytes_downloaded/download_time/1e6:.1f} MB/s")
        print(f"Output saved to: {self.output_dir}")
        print(f"\nEfficiency: Using sharding saved ~{(self.num_workers-1)/self.num_workers*100:.0f}% bandwidth!")


def main():
    """Main entry point for the fast FineWeb downloader."""
    parser = argparse.ArgumentParser(description="Fast parallel FineWeb downloader")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: download only two chunks")
    parser.add_argument("--workers", type=int, default=2,
                       help="Number of download workers (default: 2)")
    parser.add_argument("--chunk-size", type=int, default=100000,
                       help="Documents per chunk file (default: 100000)")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="Maximum documents to download (default: all)")
    parser.add_argument("--queue-size", type=int, default=100,
                       help="Maximum chunks in queue (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: data/raw/fineweb)")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("data/raw/fineweb")
    
    # Create downloader
    downloader = FineWebFastDownloader(
        output_dir=output_dir,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        queue_size=args.queue_size
    )
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nShutting down workers...")
        sys.exit(130)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start download
    downloader.download(
        max_documents=args.max_docs,
        test_mode=args.test
    )


if __name__ == "__main__":
    main()