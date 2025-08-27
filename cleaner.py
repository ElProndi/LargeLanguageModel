#!/usr/bin/env python3
"""Wikipedia article cleaner - extracts text preserving Unicode, URLs, and wiki links."""

import orjson  # Fast JSON parser (3-10x faster than standard json)
import re
import sys
import time
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
from io import StringIO

class PhaseTimer:
    """Track timing for different processing phases."""
    
    def __init__(self):
        self.phases = defaultdict(float)
        self.current_phase = None
        self.start_time = None
        self.total_time = 0
    
    def start(self, phase):
        """Start timing a phase."""
        if self.current_phase:
            self.stop()
        self.current_phase = phase
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing current phase."""
        if self.current_phase and self.start_time:
            elapsed = time.perf_counter() - self.start_time
            self.phases[self.current_phase] += elapsed
            self.total_time += elapsed
            self.current_phase = None
            self.start_time = None
    
    def get_percentages(self):
        """Get percentage breakdown of phases."""
        if self.total_time == 0:
            return {}
        return {
            phase: (time_spent / self.total_time) * 100
            for phase, time_spent in self.phases.items()
        }
    
    def get_summary(self):
        """Get formatted summary of timings."""
        percentages = self.get_percentages()
        lines = []
        for phase in sorted(self.phases.keys()):
            time_spent_ms = self.phases[phase] * 1000  # Convert to milliseconds
            pct = percentages.get(phase, 0)
            lines.append(f"  {phase:20s}: {time_spent_ms:8.1f}ms ({pct:5.1f}%)")
        return "\n".join(lines)

# Compiled regex patterns (kept for reference, but urls and wiki_links are now preserved)
PATTERNS = {
    'url': re.compile(r'https?://[^\s]+'),  # Preserved in output
    'wiki_link': re.compile(r'\[\[.*?\]\]'),  # Preserved in output
    'html': re.compile(r'<[^>]+>'),  # Removed from output
    'spaces': re.compile(r'\s+'),
    'redirect': re.compile(r'.*REDIRECT.*'),  # Only uppercase - Wikipedia directive
    'disambiguation': re.compile(r'.*may refer to:.*', re.IGNORECASE)
}

# Combined pattern for single-pass replacement - optimized for performance
# This only matches HTML tags and REDIRECT keyword, preserving URLs and wiki links
COMBINED_CLEANUP = re.compile(
    r'<[^>]+>|'                    # HTML tags
    r'REDIRECT',                   # REDIRECT keyword
    re.DOTALL                      # Allow . to match newlines
)

# Since we now preserve brackets and braces in both modes, this is identical to COMBINED_CLEANUP
COMBINED_CLEANUP_INFERENCE = COMBINED_CLEANUP

# ASCII conversion infrastructure removed - we now preserve Unicode characters

def extract_text(article):
    """Extract all text from Wikipedia article JSON using StringIO buffer"""
    # Validate article structure
    if not isinstance(article, dict):
        raise TypeError(
            f"Expected article to be a dict, got {type(article).__name__}\n"
            f"Article data may be corrupted"
        )
    
    # Use StringIO for efficient string building
    buffer = StringIO()
    first = True
    
    def extract_part(part, buf, is_first):
        """Write text parts directly to buffer."""
        if not isinstance(part, dict):
            return is_first
        
        part_type = part.get('type', '')
        
        # Extract based on type
        if part_type == 'section':
            if name := part.get('name'):
                if not is_first:
                    buf.write(' ')
                buf.write(name)
                is_first = False
            for subpart in part.get('has_parts', []):
                is_first = extract_part(subpart, buf, is_first)
                
        elif part_type in ['paragraph', 'list_item', 'field', 'image']:
            if value := part.get('value', '').strip():
                if not is_first:
                    buf.write(' ')
                buf.write(value)
                is_first = False
                
        elif part_type == 'list':
            for item in part.get('has_parts', []):
                is_first = extract_part(item, buf, is_first)
                
        elif part_type == 'infobox':
            for field in part.get('has_parts', []):
                if field.get('type') == 'field':
                    if name := field.get('name'):
                        if not is_first:
                            buf.write(' ')
                        buf.write(name)
                        is_first = False
                    if value := field.get('value'):
                        if not is_first:
                            buf.write(' ')
                        buf.write(value)
                        is_first = False
        
        return is_first
    
    # Write directly to buffer
    if abstract := article.get('abstract', '').strip():
        buffer.write(abstract)
        first = False
    
    for section in article.get('sections', []):
        first = extract_part(section, buffer, first)
    
    # Get final text
    text = buffer.getvalue()
    buffer.close()
    return text if text else None

def clean_text(text):
    """Clean text preserving Unicode, URLs, and wiki links - only removes HTML tags."""
    if not text:
        return ""
    
    # Validate input type
    if not isinstance(text, str):
        raise TypeError(
            f"Expected text to be string, got {type(text).__name__}\n"
            f"Text extraction may have failed"
        )
    
    # Check for redirect/disambiguation
    first_50 = text[:50] if len(text) > 50 else text
    if 'REDIRECT' in first_50:
        return ""
    if 'may refer to:' in first_50.lower():
        return ""
    
    # Single-pass cleanup: remove only HTML tags and REDIRECT keyword
    # Preserves: URLs, wiki links [[...]], brackets [...], braces {...}, Unicode
    text = COMBINED_CLEANUP.sub(' ', text)
    
    # Final whitespace cleanup - normalize spaces, tabs, and newlines
    text = PATTERNS['spaces'].sub(' ', text.replace('\t', ' ').replace('\n', ' '))
    text = text.strip()
    
    return text

def clean_text_for_inference(text):
    """Clean text for inference - preserves Unicode, URLs, wiki links, brackets, and braces.
    
    Since we now preserve all these elements in the main clean_text() function,
    this is now identical to clean_text(). Kept for backward compatibility.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with Unicode, URLs, and wiki links preserved
    """
    return clean_text(text)

def clean_for_tokenizer(text, preserve_braces=True):
    """Convenience wrapper for cleaning text before tokenization.
    
    This is the main entry point for external code that needs to clean text
    before passing it to the tokenizer.
    
    Args:
        text: Input text to clean
        preserve_braces: Deprecated parameter, kept for backward compatibility.
                        Braces are always preserved now.
    
    Returns:
        Cleaned text with Unicode, URLs, and wiki links preserved, ready for tokenization
    
    Example:
        >>> from cleaner import clean_for_tokenizer
        >>> # Unicode, URLs, wiki links, brackets, and braces are all preserved
        >>> clean_text = clean_for_tokenizer("Check [[Python]] at https://python.org {code}")
        >>> print(clean_text)  # "Check [[Python]] at https://python.org {code}"
    """
    # Parameter preserve_braces is deprecated - always preserve now
    return clean_text(text)

def process_file_worker(args):
    """Worker function for multiprocessing - processes a single JSONL file."""
    jsonl_path, output_dir, file_idx, max_articles_per_file = args
    
    # Convert strings back to Path objects if needed
    jsonl_path = Path(jsonl_path) if isinstance(jsonl_path, str) else jsonl_path
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    
    output_file = output_dir / f"cleaned_{file_idx}.txt"
    count = 0
    timer = PhaseTimer()
    total_chars = 0
    
    # Larger buffer for batch writing - reduces I/O overhead
    BUFFER_SIZE = 10000  # Write every 10000 articles
    write_buffer = []
    buffer_chars = 0
    BUFFER_CHAR_LIMIT = 10_000_000  # Also flush at 10MB
    
    # Return results instead of printing (for aggregation)
    result = {'file': jsonl_path.name, 'count': 0, 'chars': 0, 'timer': timer, 'error': None}
    
    start = time.time()
    
    # Open output file once for entire processing - fail on any I/O error
    try:
        out_f = open(output_file, 'w', encoding='utf-8', buffering=8388608)  # 8MB buffer
    except IOError as e:
        result['error'] = RuntimeError(
            f"Failed to open output file for writing: {output_file}\n"
            f"Error: {e}\n"
            f"Check permissions and disk space"
        )
        return result
    
    try:
        # Process file line by line - memory efficient
        timer.start('Reading')
        try:
            f = open(jsonl_path, 'r', encoding='utf-8')
        except IOError as e:
            out_f.close()  # Clean up output file
            raise RuntimeError(
                f"Failed to open input file: {jsonl_path}\n"
                f"Error: {e}\n"
                f"File may be corrupted or permissions issue"
            ) from e
        timer.stop()
        
        try:
            
            for i, line in enumerate(f, 1):
                if max_articles_per_file and count >= max_articles_per_file:
                    # Silently stop at limit (no print in worker)
                    break
                
                try:
                    # Parse JSON using orjson (much faster)
                    timer.start('JSON Parsing')
                    article = orjson.loads(line)
                    timer.stop()
                    
                    # Extract text - let any errors bubble up
                    timer.start('Text Extraction')
                    try:
                        text = extract_text(article)
                    except Exception as e:
                        timer.stop()
                        raise RuntimeError(
                            f"Text extraction failed at line {i} in {jsonl_path.name}\n"
                            f"Article title: {article.get('name', 'unknown')}\n"
                            f"Error: {e}"
                        ) from e
                    timer.stop()
                    
                    if text:
                        # Clean text - let any errors bubble up
                        timer.start('Text Cleaning')
                        try:
                            cleaned = clean_text(text)
                        except Exception as e:
                            timer.stop()
                            raise RuntimeError(
                                f"Text cleaning failed at line {i} in {jsonl_path.name}\n"
                                f"Article title: {article.get('name', 'unknown')}\n"
                                f"Text length: {len(text) if text else 0}\n"
                                f"Error: {e}"
                            ) from e
                        timer.stop()
                        
                        if cleaned and len(cleaned) > 50:  # Min length
                            write_buffer.append(cleaned)
                            cleaned_len = len(cleaned)
                            total_chars += cleaned_len
                            buffer_chars += cleaned_len
                            count += 1
                            
                            # Write buffer when full (by count or size)
                            if len(write_buffer) >= BUFFER_SIZE or buffer_chars >= BUFFER_CHAR_LIMIT:
                                timer.start('Writing')
                                try:
                                    out_f.write('\n'.join(write_buffer) + '\n')
                                except IOError as e:
                                    raise RuntimeError(
                                        f"Failed to write buffer at article {count} to {output_file}\n"
                                        f"Error: {e}\n"
                                        f"Check disk space - may be full"
                                    ) from e
                                timer.stop()
                                write_buffer.clear()
                                buffer_chars = 0
                    
                    # Progress tracking removed for worker - will be handled by main process
                        
                except (orjson.JSONDecodeError, ValueError) as e:
                    # Fail fast with clear error message
                    # orjson can raise ValueError for some invalid JSON
                    raise RuntimeError(
                        f"Failed to parse JSON at line {i} in {jsonl_path.name}: {e}"
                    ) from e
                except KeyError as e:
                    # Fail fast on missing required fields
                    raise RuntimeError(
                        f"Missing required field at line {i} in {jsonl_path.name}: {e}\n"
                        f"Article structure may be invalid"
                    ) from e
                except Exception as e:
                    # Catch any other unexpected errors and fail immediately
                    raise RuntimeError(
                        f"Unexpected error at line {i} in {jsonl_path.name}: {e}\n"
                        f"Error type: {type(e).__name__}"
                    ) from e
        
            # Write remaining buffer
            if write_buffer:
                timer.start('Writing')
                try:
                    out_f.write('\n'.join(write_buffer) + '\n')
                except IOError as e:
                    raise RuntimeError(
                        f"Failed to write final buffer to {output_file}\n"
                        f"Error: {e}\n"
                        f"Check disk space and permissions"
                    ) from e
                timer.stop()
        finally:
            f.close()
    except Exception as e:
        # Store exception in result for main process to handle
        result['error'] = e
        try:
            out_f.close()
        except:
            pass  # Already have the main error
        return result
    finally:
        out_f.close()
    
    elapsed = time.time() - start
    elapsed_ms = elapsed * 1000  # Convert to milliseconds
    
    # Update result
    result['count'] = count
    result['chars'] = total_chars
    result['timer'] = timer
    result['elapsed_ms'] = elapsed_ms
    
    return result

def main():
    """Process Wikipedia dump files preserving Unicode characters."""
    data_dir = Path("/home/andrea/Desktop/data/raw/enwiki_namespace_0")
    output_dir = Path("/home/andrea/Desktop/data/cleaned_articles")
    num_workers = 4
    
    if not data_dir.exists():
        # Fail fast with clear error about missing data directory
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Expected Wikipedia dump files at this location"
        )
    
    # Create output directory - let any permission errors bubble up
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Failed to create output directory: {output_dir}\n"
            f"Error: {e}\n"
            f"Check permissions and disk space"
        ) from e
    
    # Get all JSONL files
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        # Fail fast - no files to process is an error condition
        raise FileNotFoundError(
            f"No JSONL files found in {data_dir}\n"
            f"Expected files matching pattern: *.jsonl"
        )
    
    # Check for test mode
    test_mode = '--test' in sys.argv
    max_articles_per_file = 50000 if test_mode else None  # Reduced for faster testing
    
    print(f"\n{'='*60}")
    print(f"Wikipedia Cleaner - Unicode Preserving {'(TEST MODE)' if test_mode else '(FULL MODE)'}")
    print(f"Files: {len(jsonl_files)}")
    print(f"Workers: {num_workers} parallel processes")
    if test_mode and max_articles_per_file:
        files_in_test = min(len(jsonl_files), num_workers)
        print(f"Target: {max_articles_per_file:,} articles per file")
        print(f"Test files: {files_in_test} (matches {num_workers} workers)")
        print(f"Max total: {max_articles_per_file * files_in_test:,} articles")
    print(f"{'='*60}")
    
    # Process files in parallel
    total_articles = 0
    total_chars = 0
    global_timer = PhaseTimer()
    overall_start = time.time()
    file_stats = []  # More memory-efficient than dict for large file counts
    
    # Prepare arguments for workers
    worker_args = []
    
    if test_mode:
        # In test mode, process exactly as many files as workers
        files_to_process = min(len(jsonl_files), num_workers)
    else:
        files_to_process = len(jsonl_files)
    
    for idx, jsonl_file in enumerate(jsonl_files[:files_to_process]):
        # Convert Path objects to strings for pickling
        worker_args.append((str(jsonl_file), str(output_dir), idx, max_articles_per_file))
    
    print(f"\nProcessing {len(worker_args)} file{'s' if len(worker_args) > 1 else ''} with {num_workers} workers...")
    if test_mode and max_articles_per_file:
        print(f"Each file limited to {max_articles_per_file:,} articles")
    print(f"{'='*60}")
    
    # Process files in parallel using multiprocessing.Pool with optimized settings
    with mp.Pool(processes=num_workers, maxtasksperchild=10) as pool:
        # Use imap_unordered for better performance (order doesn't matter)
        # Each worker gets one complete file to process
        results = pool.imap_unordered(process_file_worker, worker_args)
        
        # Track progress and collect results
        completed = 0
        for result in results:
            completed += 1
            
            # Check for errors and fail fast
            if result['error']:
                pool.terminate()  # Stop all workers
                pool.join()
                raise result['error']
            
            # Update totals
            total_articles += result['count']
            total_chars += result['chars']
            
            # Update global timer
            if result['timer']:
                for phase, time_spent in result['timer'].phases.items():
                    global_timer.phases[phase] += time_spent
                global_timer.total_time += result['timer'].total_time
            
            # Track file stats
            if result['count'] > 0:
                # Find the worker args for this result
                for arg_tuple in worker_args:
                    if Path(arg_tuple[0]).name == result['file']:
                        file_idx = arg_tuple[2]
                        file_stats.append((f"cleaned_{file_idx}.txt", result['count']))
                        break
            
            # Progress update
            elapsed_so_far = time.time() - overall_start
            rate = total_articles / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"[{completed:2d}/{len(worker_args)}] {result['file']:40s} {result['count']:6,} articles in {result['elapsed_ms']:7.1f}ms | Total: {total_articles:,} @ {rate:.0f} art/s")
    
    # Save metadata efficiently
    metadata = {
        'total_articles': total_articles,
        'total_chars': total_chars,
        'files': dict(file_stats)  # Convert to dict only when saving
    }
    
    # Save metadata - fail on any write error
    metadata_file = output_dir / "metadata.json"
    try:
        with open(metadata_file, 'wb') as f:  # orjson writes bytes
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    except (IOError, TypeError) as e:
        raise RuntimeError(
            f"Failed to save metadata to {metadata_file}\n"
            f"Error: {e}\n"
            f"Processing completed but metadata not saved"
        ) from e
    
    # Calculate overall statistics
    overall_elapsed = time.time() - overall_start
    overall_elapsed_ms = overall_elapsed * 1000  # Convert to milliseconds
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {total_articles:,} articles, {total_chars:,} chars")
    print(f"Time: {overall_elapsed_ms:.1f}ms total ({overall_elapsed:.2f}s)")
    print(f"Rate: {total_articles/overall_elapsed:.0f} articles/sec" if overall_elapsed > 0 else "")
    print(f"\nPhase Timing Breakdown:")
    print(global_timer.get_summary())
    print(f"\nOutput: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        # Ensure any unhandled exception provides full context
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"FATAL ERROR - Processing failed", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nFull traceback:", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        sys.exit(1)