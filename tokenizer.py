#!/usr/bin/env python3
"""Wikipedia tokenizer training using Hugging Face tokenizers library."""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Union
import orjson

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders


class WikipediaTokenizer:
    """Fast BPE tokenizer for Wikipedia corpus."""
    
    def __init__(self, vocab_size: int = 16384):
        """Initialize tokenizer with specified vocabulary size.
        
        Args:
            vocab_size: Target vocabulary size (default 16384 for normal mode)
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.special_tokens = ["<BOS>", "<EOS>", "<UNK>"]
        
    def initialize_tokenizer(self) -> Tokenizer:
        """Create a new BPE tokenizer for ASCII-only text."""
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        
        # Use ByteLevel pre-tokenization for efficient space handling
        # This splits on whitespace while preserving spaces as special byte tokens
        # Standard approach used by GPT-2/GPT-3 for better vocabulary efficiency
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Add ByteLevel decoder to properly reconstruct spaces from byte tokens
        # This ensures perfect round-trip encoding/decoding of ASCII text
        tokenizer.decoder = decoders.ByteLevel()
        
        return tokenizer
    
    def get_training_files(self, test_mode: bool = False) -> List[str]:
        """Get list of cleaned article files for training.
        
        Args:
            test_mode: If True, only use first cleaned file
            
        Returns:
            List of file paths as strings
        """
        data_dir = Path("/home/andrea/Desktop/data/cleaned_articles")
        
        if test_mode:
            # Test mode: only use first file
            files = [str(data_dir / "cleaned_0.txt")]
        else:
            # Normal mode: use all cleaned files
            cleaned_files = sorted(data_dir.glob("cleaned_*.txt"))
            files = [str(f) for f in cleaned_files if f.suffix == ".txt"]
        
        print(f"\nPrepared {len(files)} file(s) for training")
        for f in files[:3]:  # Show first few files
            print(f"  - {Path(f).name}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more files")
        
        return files
    
    def train(self, test_mode: bool = False):
        """Train the tokenizer on the Wikipedia corpus.
        
        Args:
            test_mode: If True, use reduced vocab (256) and only first file
        """
        # Adjust vocab size for test mode
        if test_mode:
            self.vocab_size = 256
            print(f"Test mode: Vocab size = {self.vocab_size}")
        else:
            # Ensure power of 2 closest to 16000 (which is 16384 = 2^14)
            self.vocab_size = 16384
            print(f"Normal mode: Vocab size = {self.vocab_size}")
        
        # Initialize tokenizer
        self.tokenizer = self.initialize_tokenizer()
        
        # Create trainer with special tokens
        # ByteLevel pre-tokenization automatically handles the byte alphabet
        # However, we explicitly include ALL ASCII characters to ensure coverage
        # This is important for characters like { } that are removed during cleaning
        
        # Force all ASCII characters into initial alphabet
        # ByteLevel will encode these as Ġ-prefixed tokens where appropriate
        ascii_chars = [chr(i) for i in range(32, 127)]  # All printable ASCII
        
        trainer = trainers.BpeTrainer(  # type: ignore
            vocab_size=self.vocab_size,  # type: ignore
            min_frequency=2,  # type: ignore  # Minimum frequency for a pair to be merged
            special_tokens=self.special_tokens,  # type: ignore
            show_progress=True,  # type: ignore
            initial_alphabet=ascii_chars  # type: ignore  # Force all ASCII into vocab
        )
        
        print(f"\n{'='*60}")
        print(f"Training BPE Tokenizer")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Special tokens: {self.special_tokens}")
        print(f"Mode: {'TEST' if test_mode else 'NORMAL'}")
        print(f"Files: {'cleaned_0.txt only' if test_mode else 'All 38 cleaned files'}")
        print(f"{'='*60}")
        
        # Get training files and let Hugging Face handle the I/O
        start_time = time.time()
        
        # Get list of training files
        training_files = self.get_training_files(test_mode)
        
        # Train directly from files - Hugging Face handles streaming and memory management
        print(f"\nTraining tokenizer from {len(training_files)} file(s)...")
        print("Note: Hugging Face will handle file I/O and memory management internally")
        self.tokenizer.train(files=training_files, trainer=trainer)
        
        # Add post-processor for special tokens (GPT-style: BOS at start, EOS at end)
        self.tokenizer.post_processor = processors.TemplateProcessing(  # type: ignore
            single="<BOS> $A <EOS>",
            pair="<BOS> $A <EOS> $B <EOS>",  # For potential future use with paired sequences
            special_tokens=[
                ("<BOS>", self.tokenizer.token_to_id("<BOS>")),
                ("<EOS>", self.tokenizer.token_to_id("<EOS>")),
            ],
        )
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed:.1f}s")
        print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")
        print(f"{'='*60}\n")
    
    def validate_ascii(self, text: Union[str, List[str]]) -> None:
        """Validate that input text is pure ASCII.
        
        Args:
            text: Single string or list of strings to validate
            
        Raises:
            ValueError: If non-ASCII characters are found
        """
        texts_to_check = text if isinstance(text, list) else [text]
        
        for i, t in enumerate(texts_to_check):
            try:
                # This will raise UnicodeEncodeError if non-ASCII present
                t.encode('ascii')
            except UnicodeEncodeError as e:
                # Find the problematic character for better error message
                for j, char in enumerate(t):
                    if ord(char) > 127:
                        raise ValueError(
                            f"Non-ASCII character found in {'batch item ' + str(i) if isinstance(text, list) else 'text'}: "
                            f"'{char}' (Unicode {ord(char)}) at position {j}. "
                            f"All text must be pure ASCII (characters 0-127)."
                        ) from e
    
    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True) -> Union[List[int], List[List[int]]]:
        """Fast encoding of text to token IDs using the fastest available methods.
        
        Args:
            text: Single string or list of strings to encode
            add_special_tokens: Whether to add special tokens (default True)
            
        Returns:
            Token IDs as list of ints (single) or list of lists (batch)
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not trained. Run train() first.")
        
        # Validate ASCII-only constraint
        self.validate_ascii(text)
        
        # Check if batch or single
        is_batch = isinstance(text, list)
        
        if is_batch:
            # Use encode_batch_fast for maximum speed - skips offset tracking
            # This is faster than encode_batch since we don't need character offsets
            encodings = self.tokenizer.encode_batch_fast(text, add_special_tokens=add_special_tokens)
            # Ensure each encoding returns a list
            return [list(enc.ids) if not isinstance(enc.ids, list) else enc.ids for enc in encodings]
        else:
            # Single encoding - already optimized
            encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            # Ensure we return a list (encoding.ids might be tuple or other type)
            ids = encoding.ids
            if not isinstance(ids, list):
                ids = list(ids)
            return ids
    
    def decode(self, ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """Fast decoding of token IDs to text using optimized Rust implementation.
        
        Args:
            ids: Token IDs as list of ints (single) or list of lists (batch)
            skip_special_tokens: Whether to skip special tokens (default True)
            
        Returns:
            Decoded text as string (single) or list of strings (batch)
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not trained. Run train() first.")
        
        # Check if batch or single
        is_batch = isinstance(ids[0], list) if ids else False
        
        if is_batch:
            # Batch decoding
            return self.tokenizer.decode_batch(ids, skip_special_tokens=skip_special_tokens)
        else:
            # Single decoding
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def save(self, path: str):
        """Save trained tokenizer to disk.
        
        Args:
            path: Directory path to save tokenizer
        """
        if not self.tokenizer:
            raise RuntimeError("No tokenizer to save. Train first.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        tokenizer_file = save_path / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_file))
        
        # Save metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "actual_vocab_size": self.tokenizer.get_vocab_size()
        }
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, 'wb') as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
        
        print(f"Tokenizer saved to {save_path}")
    
    def load(self, path: str):
        """Load trained tokenizer from disk.
        
        Args:
            path: Directory path containing saved tokenizer
        """
        load_path = Path(path)
        tokenizer_file = load_path / "tokenizer.json"
        
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")
        
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        
        # Load metadata
        metadata_file = load_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = orjson.loads(f.read())
                self.vocab_size = metadata["vocab_size"]
                self.special_tokens = metadata["special_tokens"]
        
        print(f"Tokenizer loaded from {load_path}")
        print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")


def main():
    """Main entry point for tokenizer training and usage."""
    parser = argparse.ArgumentParser(description="Wikipedia Tokenizer Training")
    parser.add_argument("--test", action="store_true", 
                       help="Test mode: use first file only with vocab=256")
    parser.add_argument("--encode", type=str, 
                       help="Encode text using trained tokenizer")
    parser.add_argument("--decode", type=str, 
                       help="Decode token IDs (comma-separated)")
    parser.add_argument("--save", type=str, 
                       help="Save trained tokenizer to directory")
    parser.add_argument("--load", type=str, 
                       help="Load trained tokenizer from directory")
    
    args = parser.parse_args()
    
    # Create tokenizer instance
    tokenizer = WikipediaTokenizer()
    
    # Determine default save/load paths
    default_path = "tokenizers/test_tokenizer" if args.test else "tokenizers/full_tokenizer"
    
    # Load existing or train new
    if args.load:
        tokenizer.load(args.load)
    elif args.encode or args.decode:
        # Try to load default tokenizer for the mode
        try:
            tokenizer.load(default_path)
        except FileNotFoundError:
            print(f"No trained tokenizer found at {default_path}")
            print("Training new tokenizer...")
            tokenizer.train(test_mode=args.test)
            tokenizer.save(default_path)
    else:
        # Train new tokenizer
        tokenizer.train(test_mode=args.test)
        
        # Save to default or specified path
        save_path = args.save if args.save else default_path
        tokenizer.save(save_path)
    
    # Perform encoding/decoding if requested
    if args.encode:
        print(f"\nEncoding: '{args.encode}'")
        ids = tokenizer.encode(args.encode)
        print(f"Token IDs: {ids}")
        print(f"Number of tokens: {len(ids)}")
        
        # Also show decoding for verification
        decoded = tokenizer.decode(ids)
        print(f"Decoded: '{decoded}'")
    
    if args.decode:
        # Parse comma-separated IDs
        ids = [int(x.strip()) for x in args.decode.split(",")]
        print(f"\nDecoding IDs: {ids}")
        text = tokenizer.decode(ids)
        print(f"Text: '{text}'")
    
    # If no specific action, show sample encodings with various ASCII characters
    if not (args.encode or args.decode):
        print(f"\n{'='*60}")
        print("Sample Encodings with Various ASCII Characters")
        print(f"{'='*60}")
        
        # Test samples with different ASCII characters
        test_samples = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, World! How are you today?",
            "Testing 123... Special chars: @#$%^&*()",
            "Numbers: 0123456789 Math: 2+2=4, 10-5=5, 3*4=12",
            "Punctuation: . , ; : ' \" ! ? - ( ) [ ] { }",
            "    Leading spaces and     multiple     spaces    ",
            "Mixed: ABC xyz 123 !@# \"quoted\" (parentheses)",
            "Email: user@example.com URL: https://www.example.org",
            "Path: /home/user/file.txt C:\\Windows\\System32",
            "Code: if (x > 0) { return x * 2; } else { return -1; }",
            "Math symbols: + - * / = < > <= >= != == && ||",
            "Escaped: \\n \\t \\r \\\\ Quote: 'single' \"double\"",
        ]
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\nSample {i}:")
            print(f"Original: '{sample}'")
            print(f"Length:   {len(sample)} characters")
            
            # Encode
            ids = tokenizer.encode(sample, add_special_tokens=False)  # Don't add CLS/SEP for clearer testing
            print(f"Token IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")  # Show first 20 tokens
            print(f"Tokens:    {len(ids)} tokens")
            
            # Decode
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            print(f"Decoded:  '{decoded}'")
            
            # Verify exact match
            if decoded == sample:
                print(f"✓ Perfect reconstruction")
            else:
                print(f"✗ Mismatch in reconstruction")
                if len(decoded) != len(sample):
                    print(f"  Length difference: original={len(sample)}, decoded={len(decoded)}")
        
        print(f"\n{'='*60}")
        print("Tokenization Statistics")
        print(f"{'='*60}")
        if tokenizer.tokenizer:
            print(f"Vocabulary size: {tokenizer.tokenizer.get_vocab_size()}")
            print(f"Compression ratio: ~{sum(len(s) for s in test_samples) / sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in test_samples):.2f}x")
        else:
            print("Tokenizer not initialized")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTokenizer training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)