#!/usr/bin/env python3
"""CodeLlama tokenizer wrapper for the LLM training pipeline."""

import sys
import argparse
from pathlib import Path
from typing import List, Union, Optional
import orjson

# Try to import transformers
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers library not found. Please install it with:")
    print("  pip install transformers")
    sys.exit(1)


class WikipediaTokenizer:
    """Wrapper for CodeLlama tokenizer from HuggingFace."""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-hf"):
        """Initialize with CodeLlama tokenizer.
        
        Args:
            model_name: HuggingFace model name for CodeLlama tokenizer
        """
        self.model_name = model_name
        self.tokenizer = None
        self.vocab_size = None
        
    def train(self, test_mode: bool = False):
        """Load pre-trained CodeLlama tokenizer.
        
        Note: This doesn't actually train a tokenizer, it loads the pre-trained one.
        We keep this method for compatibility with the existing pipeline.
        
        Args:
            test_mode: Ignored - we always use the full CodeLlama tokenizer
        """
        print(f"\n{'='*60}")
        print(f"Loading CodeLlama Tokenizer")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")
        
        try:
            # Load pre-trained tokenizer from HuggingFace
            print(f"Downloading/loading tokenizer from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,  # Use fast Rust-based tokenizer
                trust_remote_code=False
            )
            
            # Get vocab size
            self.vocab_size = len(self.tokenizer)
            
            # Set padding token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"\n{'='*60}")
            print(f"Tokenizer loaded successfully!")
            print(f"Vocabulary size: {self.vocab_size}")
            print(f"Special tokens:")
            print(f"  BOS: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id})")
            print(f"  EOS: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
            print(f"  UNK: {self.tokenizer.unk_token} (ID: {self.tokenizer.unk_token_id})")
            print(f"  PAD: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Make sure you have internet connection for downloading the tokenizer.")
            raise
    
    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True) -> Union[List[int], List[List[int]]]:
        """Encode text to token IDs using CodeLlama tokenizer.
        
        Args:
            text: Single string or list of strings to encode
            add_special_tokens: Whether to add special tokens (default True)
            
        Returns:
            Token IDs as list of ints (single) or list of lists (batch)
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Run train() first.")
        
        # Check if batch or single
        is_batch = isinstance(text, list)
        
        if is_batch:
            # Batch encoding
            encodings = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_tensors=None
            )
            return encodings["input_ids"]
        else:
            # Single encoding
            encoding = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False
            )
            return encoding
    
    def decode(self, ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """Decode token IDs to text using CodeLlama tokenizer.
        
        Args:
            ids: Token IDs as list of ints (single) or list of lists (batch)
            skip_special_tokens: Whether to skip special tokens (default True)
            
        Returns:
            Decoded text as string (single) or list of strings (batch)
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Run train() first.")
        
        # Check if batch or single
        is_batch = isinstance(ids[0], list) if ids else False
        
        if is_batch:
            # Batch decoding
            return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)
        else:
            # Single decoding
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def save(self, path: str):
        """Save tokenizer to disk.
        
        Args:
            path: Directory path to save tokenizer
        """
        if not self.tokenizer:
            raise RuntimeError("No tokenizer to save. Load first.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save HuggingFace tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save metadata for compatibility
        metadata = {
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "bos_token": self.tokenizer.bos_token,
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token": self.tokenizer.eos_token,
                "eos_token_id": self.tokenizer.eos_token_id,
                "unk_token": self.tokenizer.unk_token,
                "unk_token_id": self.tokenizer.unk_token_id,
                "pad_token": self.tokenizer.pad_token,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
        }
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, 'wb') as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
        
        print(f"Tokenizer saved to {save_path}")
    
    def load(self, path: str):
        """Load tokenizer from disk.
        
        Args:
            path: Directory path containing saved tokenizer
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {load_path}")
        
        # Check if it's a HuggingFace tokenizer directory
        if (load_path / "tokenizer_config.json").exists():
            # Load HuggingFace tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)
            self.vocab_size = len(self.tokenizer)
            
            # Load metadata if available
            metadata_file = load_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata = orjson.loads(f.read())
                    self.model_name = metadata.get("model_name", "codellama/CodeLlama-7b-hf")
            
            print(f"Tokenizer loaded from {load_path}")
            print(f"Vocabulary size: {self.vocab_size}")
        else:
            raise ValueError(f"Invalid tokenizer directory: {load_path}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size of the tokenizer.
        
        Returns:
            Vocabulary size
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded.")
        return len(self.tokenizer)
    
    def get_special_token_ids(self) -> dict:
        """Get special token IDs.
        
        Returns:
            Dictionary with special token IDs
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded.")
        
        return {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }


def main():
    """Main entry point for tokenizer usage."""
    parser = argparse.ArgumentParser(description="CodeLlama Tokenizer for LLM Pipeline")
    parser.add_argument("--test", action="store_true", 
                       help="Test mode (ignored - always uses full tokenizer)")
    parser.add_argument("--encode", type=str, 
                       help="Encode text using tokenizer")
    parser.add_argument("--decode", type=str, 
                       help="Decode token IDs (comma-separated)")
    parser.add_argument("--save", type=str, default="tokenizers/codellama_tokenizer",
                       help="Save tokenizer to directory")
    parser.add_argument("--load", type=str, 
                       help="Load tokenizer from directory")
    
    args = parser.parse_args()
    
    # Create tokenizer instance
    tokenizer = WikipediaTokenizer()
    
    # Default save path
    default_path = "tokenizers/codellama_tokenizer"
    
    # Load existing or download new
    if args.load:
        tokenizer.load(args.load)
    elif args.encode or args.decode:
        # Try to load saved tokenizer first
        try:
            tokenizer.load(default_path)
        except FileNotFoundError:
            print(f"No saved tokenizer found at {default_path}")
            print("Downloading CodeLlama tokenizer from HuggingFace...")
            tokenizer.train()
            tokenizer.save(default_path)
    else:
        # Download and save tokenizer
        tokenizer.train(test_mode=args.test)
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
    
    # If no specific action, show sample encodings
    if not (args.encode or args.decode):
        print(f"\n{'='*60}")
        print("Sample Encodings")
        print(f"{'='*60}")
        
        # Test samples - now including Unicode and code
        test_samples = [
            "Hello, World! How are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Testing 123... Special chars: @#$%^&*()",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "print('Hello, 世界!')",  # Unicode test
            "for i in range(10):\n    print(f'Number: {i}')",
            "class MyClass:\n    def __init__(self):\n        self.value = 42",
            "// This is a C++ comment\nint main() { return 0; }",
        ]
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\nSample {i}:")
            print(f"Original: {sample!r}")
            print(f"Length:   {len(sample)} characters")
            
            # Encode
            ids = tokenizer.encode(sample, add_special_tokens=False)
            print(f"Token IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
            print(f"Tokens:    {len(ids)} tokens")
            print(f"Compression: {len(sample)/len(ids):.2f}x")
            
            # Decode
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            print(f"Decoded:  {decoded!r}")
            
            # Verify exact match
            if decoded == sample:
                print(f"✓ Perfect reconstruction")
            else:
                print(f"✗ Mismatch in reconstruction")
        
        print(f"\n{'='*60}")
        print("Tokenization Statistics")
        print(f"{'='*60}")
        print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        special_tokens = tokenizer.get_special_token_ids()
        print(f"Special token IDs:")
        for name, id_val in special_tokens.items():
            print(f"  {name}: {id_val}")
        
        avg_compression = sum(len(s) / len(tokenizer.encode(s, add_special_tokens=False)) 
                             for s in test_samples) / len(test_samples)
        print(f"Average compression ratio: {avg_compression:.2f}x")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTokenizer operation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)