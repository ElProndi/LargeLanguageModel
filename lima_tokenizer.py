#!/usr/bin/env python3
"""LIMA dataset tokenizer for instruction fine-tuning."""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
from tqdm import tqdm
import orjson

# Import the existing tokenizer from your codebase
from tokenizer import WikipediaTokenizer


class LIMATokenizer:
    """Tokenizer for LIMA instruction dataset."""
    
    def __init__(self, max_length: int = 512):
        """Initialize LIMA tokenizer.
        
        Args:
            max_length: Maximum sequence length in tokens (default 512)
        """
        self.max_length = max_length
        self.tokenizer = None
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.bos_token_id = None
        self.eos_token_id = None
        
        # Statistics tracking
        self.total_examples = 0
        self.filtered_examples = 0
        self.total_tokens = 0
        self.token_distribution = []
        
    def load_tokenizer(self):
        """Load CodeLlama tokenizer from saved directory."""
        tokenizer_path = Path("tokenizers/codellama_tokenizer")
        
        # Create WikipediaTokenizer instance (wrapper for CodeLlama)
        self.tokenizer = WikipediaTokenizer()
        
        # Load the saved tokenizer
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Please run tokenizer.py first to download/save the CodeLlama tokenizer."
            )
        
        print(f"Loading CodeLlama tokenizer from {tokenizer_path}")
        self.tokenizer.load(str(tokenizer_path))
        
        # Get special token IDs
        special_tokens = self.tokenizer.get_special_token_ids()
        self.bos_token_id = special_tokens["bos_token_id"]
        self.eos_token_id = special_tokens["eos_token_id"]
        
        print(f"Tokenizer loaded successfully")
        print(f"  Vocabulary size: {self.tokenizer.get_vocab_size()}")
        print(f"  BOS token: {self.bos_token} (ID: {self.bos_token_id})")
        print(f"  EOS token: {self.eos_token} (ID: {self.eos_token_id})")
        print()
        
    def format_conversation(self, conversations: List[str]) -> str:
        """Format a conversation with role markers and special tokens.
        
        For each turn:
        - User messages start with "<s>User: "
        - Assistant messages start with "Assistant: " and end with "</s>"
        
        Args:
            conversations: List of conversation turns (alternating user/assistant)
            
        Returns:
            Formatted conversation string
        """
        formatted_parts = []
        
        for i, turn in enumerate(conversations):
            if i % 2 == 0:  # User turn (even indices)
                # Add BOS token and User marker
                formatted_parts.append(f"{self.bos_token}User: {turn}")
            else:  # Assistant turn (odd indices)
                # Add Assistant marker and EOS token
                formatted_parts.append(f"Assistant: {turn}{self.eos_token}")
        
        # Join all parts with newlines for readability
        # The newlines will be tokenized as well, providing natural separation
        return "\n".join(formatted_parts)
    
    def tokenize_example(self, formatted_text: str) -> List[int]:
        """Tokenize a formatted conversation.
        
        Args:
            formatted_text: Formatted conversation string with role markers
            
        Returns:
            List of token IDs
        """
        # Tokenize without adding special tokens (we've already added them manually)
        token_ids = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        return token_ids
    
    def process_dataset(self, input_path: str, output_dir: str, batch_size: int = 100):
        """Process the LIMA dataset and create tokenized sequences.
        
        Args:
            input_path: Path to LIMA jsonl file
            output_dir: Directory to save tokenized data
            batch_size: Number of examples to process at once
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"\n{'='*60}")
        print(f"LIMA Dataset Tokenization")
        print(f"{'='*60}")
        print(f"Input: {input_path}")
        print(f"Output: {output_dir}")
        print(f"Max length: {self.max_length} tokens")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
        
        # Load all examples
        print("Loading LIMA dataset...")
        examples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        print(f"Loaded {len(examples)} examples")
        
        # Process examples in batches
        all_tokenized = []
        all_metadata = []
        
        print("\nTokenizing conversations...")
        for batch_start in tqdm(range(0, len(examples), batch_size), desc="Batches"):
            batch_end = min(batch_start + batch_size, len(examples))
            batch = examples[batch_start:batch_end]
            
            for example in batch:
                # Format the conversation
                formatted_text = self.format_conversation(example['conversations'])
                
                # Tokenize
                token_ids = self.tokenize_example(formatted_text)
                
                # Track statistics
                self.total_examples += 1
                token_count = len(token_ids)
                self.token_distribution.append(token_count)
                
                # Filter by length
                if token_count <= self.max_length:
                    all_tokenized.append(token_ids)
                    all_metadata.append({
                        'source': example['source'],
                        'num_turns': len(example['conversations']),
                        'token_count': token_count
                    })
                    self.total_tokens += token_count
                else:
                    self.filtered_examples += 1
        
        print(f"\nTokenization complete!")
        print(f"  Total examples: {self.total_examples}")
        print(f"  Kept examples: {len(all_tokenized)}")
        print(f"  Filtered (>{self.max_length} tokens): {self.filtered_examples}")
        
        if len(all_tokenized) == 0:
            print("WARNING: No examples passed the length filter!")
            return
        
        # Pad sequences to max_length for uniform shape
        print("\nPadding sequences to uniform length...")
        padded_sequences = np.full((len(all_tokenized), self.max_length), 
                                   self.tokenizer.tokenizer.pad_token_id, 
                                   dtype=np.int16)
        
        for i, seq in enumerate(all_tokenized):
            padded_sequences[i, :len(seq)] = seq
        
        # Save tokenized data
        output_path = output_dir / "tokenized_examples.npy"
        np.save(output_path, padded_sequences)
        print(f"Saved tokenized data to: {output_path}")
        
        # Calculate and save metadata
        metadata = {
            'total_examples': self.total_examples,
            'kept_examples': len(all_tokenized),
            'filtered_examples': self.filtered_examples,
            'max_length': self.max_length,
            'total_tokens': self.total_tokens,
            'avg_tokens': self.total_tokens / len(all_tokenized) if all_tokenized else 0,
            'min_tokens': min(self.token_distribution),
            'max_tokens': max(self.token_distribution),
            'median_tokens': int(np.median(self.token_distribution)),
            'vocab_size': self.tokenizer.get_vocab_size(),
            'special_tokens': {
                'bos_token': self.bos_token,
                'bos_token_id': int(self.bos_token_id),
                'eos_token': self.eos_token,
                'eos_token_id': int(self.eos_token_id)
            },
            'sources_distribution': {}
        }
        
        # Count sources
        for meta in all_metadata:
            source = meta['source']
            metadata['sources_distribution'][source] = metadata['sources_distribution'].get(source, 0) + 1
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'wb') as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
        print(f"Saved metadata to: {metadata_path}")
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Dataset Statistics")
        print(f"{'='*60}")
        print(f"Token distribution:")
        print(f"  Min tokens: {metadata['min_tokens']}")
        print(f"  Max tokens: {metadata['max_tokens']}")
        print(f"  Median tokens: {metadata['median_tokens']}")
        print(f"  Avg tokens: {metadata['avg_tokens']:.1f}")
        print(f"\nSource distribution:")
        for source, count in sorted(metadata['sources_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
            percentage = count / len(all_tokenized) * 100
            print(f"  {source:20s}: {count:4d} ({percentage:5.1f}%)")
        print(f"{'='*60}\n")
        
        # Save a few examples for inspection
        print("Saving example conversations for inspection...")
        examples_to_save = min(5, len(all_tokenized))
        inspection_data = []
        
        for i in range(examples_to_save):
            original_example = examples[i]
            formatted = self.format_conversation(original_example['conversations'])
            tokens = all_tokenized[i]
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
            
            inspection_data.append({
                'example_id': i,
                'source': original_example['source'],
                'original_conversations': original_example['conversations'],
                'formatted_text': formatted,
                'token_ids': tokens[:50],  # First 50 tokens for brevity
                'token_count': len(tokens),
                'decoded_text': decoded
            })
        
        inspection_path = output_dir / "example_conversations.json"
        with open(inspection_path, 'wb') as f:
            f.write(orjson.dumps(inspection_data, option=orjson.OPT_INDENT_2))
        print(f"Saved example conversations to: {inspection_path}")


def main():
    """Main entry point for LIMA tokenization."""
    parser = argparse.ArgumentParser(description="Tokenize LIMA dataset for instruction fine-tuning")
    parser.add_argument("--input", type=str, 
                       default="/home/andrea/Desktop/data/post-training/lima_train.jsonl",
                       help="Path to LIMA jsonl file")
    parser.add_argument("--output", type=str,
                       default="/home/andrea/Desktop/data/post-training",
                       help="Output directory for tokenized data")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length in tokens")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Create tokenizer instance
    tokenizer = LIMATokenizer(max_length=args.max_length)
    
    # Load CodeLlama tokenizer
    tokenizer.load_tokenizer()
    
    # Process dataset
    tokenizer.process_dataset(
        input_path=args.input,
        output_dir=args.output,
        batch_size=args.batch_size
    )
    
    print("✓ LIMA dataset tokenization complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTokenization interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)