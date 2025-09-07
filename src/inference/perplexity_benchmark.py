#!/usr/bin/env python3
"""Perplexity benchmark module for evaluating language models on held-out FineWeb data.

This module evaluates model perplexity on the reserved evaluation dataset (tokens_0.npy)
which is never seen during training, providing a measure of how well the model predicts
unseen text from the same distribution.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import math

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import model utilities
try:
    # Try relative imports (when imported as a module)
    from ..dataset_preparation.tokenizer import CodeLlamaTokenizer
    from ..utils.model import TransformerLM
except ImportError:
    # Fall back to absolute imports (when run directly)
    from src.dataset_preparation.tokenizer import CodeLlamaTokenizer
    from src.utils.model import TransformerLM


def _load_tokenizer() -> CodeLlamaTokenizer:
    """Load or download the CodeLlama tokenizer."""
    print("Loading tokenizer...")
    tokenizer = CodeLlamaTokenizer()
    tokenizer_path = Path("tokenizers/codellama_tokenizer")
    
    if tokenizer_path.exists():
        try:
            tokenizer.load(str(tokenizer_path))
            print(f"CodeLlama tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            raise
    else:
        print("CodeLlama tokenizer not found locally. Downloading from HuggingFace...")
        tokenizer.train()
        tokenizer.save(str(tokenizer_path))
        print(f"CodeLlama tokenizer downloaded and saved to {tokenizer_path}")
    
    return tokenizer


def load_model_from_checkpoint(checkpoint_path: Union[str, Path], device: torch.device) -> Tuple[TransformerLM, CodeLlamaTokenizer]:
    """Load model from checkpoint with proper architecture handling."""
    checkpoint_path = Path(checkpoint_path)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config')
    if not config:
        with open("config.json", 'r') as f:
            config = json.load(f)
    
    if 'models' in config:
        model_size = checkpoint.get('model_size')
        if not model_size:
            available_sizes = list(config['models'].keys())
            model_size = available_sizes[0]
        model_config = config['models'][model_size]
    else:
        model_config = config['model']
    
    tokenizer_config = config.get('tokenizer', {})
    rope_config = config.get('rope', {})
    architecture_features = config.get('architecture_features', {})
    
    state_dict = checkpoint['model_state_dict']
    intermediate_size = None
    
    for key in state_dict.keys():
        if 'ffn.w_fused.weight' in key:
            intermediate_size = state_dict[key].shape[0] // 2
            break
        elif 'ffn.w_gate.weight' in key:
            intermediate_size = state_dict[key].shape[0]
            break
    
    print(f"  Flash Attention: ✓ Always Enabled")
    print(f"  Compilation: ✗ Disabled (benchmark mode)")
    
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        intermediate_size=intermediate_size,
        max_position_embeddings=model_config['max_position_embeddings'],
        rope_theta=rope_config.get('theta', 10000.0),
        rope_scaling=rope_config.get('scaling_factor', 1.0),
        tie_embeddings=architecture_features.get('tie_embeddings', True),
        dropout=0.0,
        attention_dropout=0.0,
        layer_norm_eps=model_config['layer_norm_eps'],
        initializer_range=model_config['initializer_range'],
        pad_token_id=tokenizer_config.get('pad_token_id', 2)
    )
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            raise RuntimeError(f"Model architecture doesn't match checkpoint: {e}")
        else:
            raise e
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")
    
    tokenizer = _load_tokenizer()
    
    return model, tokenizer


class PerplexityBenchmark:
    """Handles perplexity evaluation on held-out FineWeb data.
    
    Perplexity measures how well a probability model predicts a sample.
    Lower perplexity indicates better performance.
    """
    
    def __init__(self, device: torch.device = None, verbose: bool = True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.eval_data_path = Path("data/tokenized_datasets/fineweb_full_dataset/tokens_0.npy")
        
    def load_evaluation_data(self, max_sequences: Optional[int] = None) -> np.ndarray:
        """Load the held-out evaluation dataset (tokens_0.npy).
        
        Args:
            max_sequences: Maximum number of sequences to load (None for all)
            
        Returns:
            Numpy array of token sequences
        """
        if not self.eval_data_path.exists():
            raise FileNotFoundError(f"Evaluation data not found at {self.eval_data_path}")
        
        if self.verbose:
            print(f"Loading evaluation data from {self.eval_data_path}...")
        
        # Load data (memory-mapped for efficiency)
        data = np.load(self.eval_data_path, mmap_mode='r')
        
        if max_sequences is not None:
            data = np.array(data[:max_sequences])  # Convert to regular array for slicing
        else:
            data = np.array(data)  # Load all data
        
        if self.verbose:
            print(f"  Loaded {len(data):,} sequences")
            print(f"  Sequence length: {data.shape[1]} tokens")
            data_gb = data.nbytes / (1024**3)
            print(f"  Data size: {data_gb:.2f} GB")
        
        return data
    
    def calculate_perplexity(
        self,
        model: torch.nn.Module,
        sequences: np.ndarray,
        batch_size: int = 32,
        stride: Optional[int] = None,
        max_length: int = 2048
    ) -> Dict[str, float]:
        """Calculate perplexity on the given sequences.
        
        Args:
            model: The model to evaluate
            sequences: Token sequences to evaluate on
            batch_size: Batch size for evaluation
            stride: Stride for sliding window (None = no sliding window)
            max_length: Maximum sequence length to process
            
        Returns:
            Dictionary with perplexity metrics
        """
        model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        sequence_perplexities = []
        
        # Process in batches
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Computing perplexity"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(sequences))
                batch_sequences = sequences[start_idx:end_idx]
                
                # Convert to tensor and move to device
                input_ids = torch.tensor(batch_sequences, dtype=torch.long, device=self.device)
                
                # Handle sequences longer than max_length with sliding window
                if stride is not None and input_ids.shape[1] > max_length:
                    # Sliding window approach
                    seq_loss = 0.0
                    seq_tokens = 0
                    
                    for start_pos in range(0, input_ids.shape[1] - 1, stride):
                        end_pos = min(start_pos + max_length, input_ids.shape[1])
                        window_input = input_ids[:, start_pos:end_pos]
                        
                        # Forward pass
                        outputs = model(window_input, labels=window_input, return_dict=True)
                        
                        # Accumulate loss
                        seq_loss += outputs['loss'].item() * (end_pos - start_pos - 1)
                        seq_tokens += end_pos - start_pos - 1
                        
                        if end_pos >= input_ids.shape[1]:
                            break
                    
                    if seq_tokens > 0:
                        avg_loss = seq_loss / seq_tokens
                        total_loss += seq_loss
                        total_tokens += seq_tokens
                        sequence_perplexities.append(math.exp(avg_loss))
                else:
                    # Process entire sequence at once
                    # Truncate if necessary
                    if input_ids.shape[1] > max_length:
                        input_ids = input_ids[:, :max_length]
                    
                    # Forward pass
                    outputs = model(input_ids, labels=input_ids, return_dict=True)
                    
                    # Get loss (already averaged over tokens)
                    loss = outputs['loss'].item()
                    
                    # Calculate number of prediction tokens (all except first)
                    num_pred_tokens = (input_ids.shape[1] - 1) * input_ids.shape[0]
                    
                    # Accumulate
                    total_loss += loss * num_pred_tokens
                    total_tokens += num_pred_tokens
                    
                    # Per-sequence perplexity (for batch, take average)
                    sequence_perplexities.extend([math.exp(loss)] * input_ids.shape[0])
        
        # Calculate overall perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        
        # Calculate statistics
        results = {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'num_sequences': len(sequences),
            'median_perplexity': float(np.median(sequence_perplexities)),
            'std_perplexity': float(np.std(sequence_perplexities)),
            'min_perplexity': float(np.min(sequence_perplexities)),
            'max_perplexity': float(np.max(sequence_perplexities))
        }
        
        return results
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        max_sequences: Optional[int] = 1000,
        batch_size: int = 32,
        stride: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate a single model's perplexity on FineWeb evaluation data.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer (unused but kept for API consistency)
            max_sequences: Maximum sequences to evaluate on
            batch_size: Batch size for evaluation
            stride: Stride for sliding window
            
        Returns:
            Dictionary with evaluation results
        """
        # Load evaluation data
        sequences = self.load_evaluation_data(max_sequences)
        
        # Calculate perplexity
        if self.verbose:
            print("\nCalculating perplexity on FineWeb evaluation set...")
        
        results = self.calculate_perplexity(
            model=model,
            sequences=sequences,
            batch_size=batch_size,
            stride=stride
        )
        
        if self.verbose:
            print(f"\nResults:")
            print(f"  Perplexity: {results['perplexity']:.2f}")
            print(f"  Average Loss: {results['avg_loss']:.4f}")
            print(f"  Total Tokens: {results['total_tokens']:,}")
            print(f"  Median Perplexity: {results['median_perplexity']:.2f}")
            print(f"  Std Perplexity: {results['std_perplexity']:.2f}")
        
        return results
    
    def run_benchmark(
        self,
        checkpoint_paths: Union[List[Union[str, Path]], Dict[str, Union[str, Path]]],
        max_sequences: Optional[int] = 1000,
        batch_size: int = 32,
        stride: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run perplexity benchmark on multiple models.
        
        Args:
            checkpoint_paths: Paths to model checkpoints
            max_sequences: Maximum sequences to evaluate
            batch_size: Batch size for evaluation
            stride: Stride for sliding window
            
        Returns:
            Dictionary mapping model names to results
        """
        # Normalize input to dict format
        if isinstance(checkpoint_paths, list):
            checkpoint_dict = {Path(p).stem: p for p in checkpoint_paths}
        else:
            checkpoint_dict = checkpoint_paths
        
        results = {}
        
        print("\n" + "="*60)
        print("PERPLEXITY BENCHMARK - FineWeb Evaluation Set")
        print("="*60)
        
        for name, checkpoint_path in checkpoint_dict.items():
            print(f"\nEvaluating: {name}")
            print("-"*40)
            
            try:
                # Load model
                model, tokenizer = load_model_from_checkpoint(checkpoint_path, self.device)
                
                # Evaluate
                model_results = self.evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    max_sequences=max_sequences,
                    batch_size=batch_size,
                    stride=stride
                )
                
                results[name] = model_results
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results[name] = {'error': str(e)}
            
            # Clean up
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
        
        return results


def format_perplexity_results(results: Dict[str, Dict[str, Any]]) -> str:
    """Format perplexity benchmark results as a readable table."""
    output = []
    output.append("\n" + "="*80)
    output.append("PERPLEXITY BENCHMARK RESULTS - FineWeb Held-Out Data")
    output.append("="*80)
    output.append("\nPerplexity measures how well a model predicts unseen text.")
    output.append("Lower values indicate better performance.")
    output.append("-"*80)
    
    # Find best perplexity
    best_model = None
    best_perplexity = float('inf')
    for model_name, scores in results.items():
        if 'perplexity' in scores and scores['perplexity'] < best_perplexity:
            best_perplexity = scores['perplexity']
            best_model = model_name
    
    # Create results table
    output.append(f"\n{'Model':<30} | {'Perplexity':>12} | {'Loss':>8} | {'Sequences':>10}")
    output.append("-"*80)
    
    # Add results for each model
    for model_name, scores in results.items():
        if 'error' in scores:
            output.append(f"{model_name[:30]:<30} | {'ERROR':>12} | {'-':>8} | {'-':>10}")
        elif 'perplexity' in scores:
            ppl = scores['perplexity']
            loss = scores['avg_loss']
            num_seq = scores['num_sequences']
            
            # Format perplexity with best marker
            ppl_str = f"{ppl:.2f}"
            if model_name == best_model:
                ppl_str += " *"
            
            output.append(f"{model_name[:30]:<30} | {ppl_str:>12} | {loss:>8.4f} | {num_seq:>10}")
    
    output.append("\n" + "="*80)
    output.append("INTERPRETATION:")
    output.append("  * indicates best performing model (lowest perplexity)")
    output.append("  Perplexity < 20: Excellent")
    output.append("  Perplexity 20-50: Good")
    output.append("  Perplexity 50-100: Moderate")
    output.append("  Perplexity > 100: Poor")
    output.append("="*80)
    
    # Add summary if multiple models
    if len(results) > 1 and best_model:
        output.append("\n" + "="*80)
        output.append("SUMMARY:")
        output.append(f"  Best Model: {best_model}")
        output.append(f"  Best Perplexity: {best_perplexity:.2f}")
        
        # Calculate statistics
        perplexities = [s['perplexity'] for s in results.values() if 'perplexity' in s]
        if perplexities:
            avg_ppl = np.mean(perplexities)
            std_ppl = np.std(perplexities)
            output.append(f"  Average Perplexity: {avg_ppl:.2f}")
            output.append(f"  Standard Deviation: {std_ppl:.2f}")
        output.append("="*80)
    
    return "\n".join(output)


def save_perplexity_results(results: Dict[str, Dict[str, Any]], output_dir: str = "benchmark_results"):
    """Save perplexity benchmark results to JSON with timestamp."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"perplexity_benchmark_{timestamp}.json"
    
    # Add metadata
    full_results = {
        'benchmark': 'FineWeb_Perplexity',
        'timestamp': timestamp,
        'dataset': 'tokens_0.npy (held-out evaluation set)',
        'results': results,
        'summary': {
            'num_models': len(results),
            'metric': 'perplexity',
            'description': 'Perplexity on held-out FineWeb evaluation data'
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    # Also save formatted text version
    text_filename = output_path / f"perplexity_benchmark_{timestamp}.txt"
    with open(text_filename, 'w') as f:
        f.write(format_perplexity_results(results))
    
    print(f"Text summary saved to: {text_filename}")


# Main execution for standalone testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run perplexity benchmark on FineWeb evaluation data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--max-sequences", type=int, default=1000,
                        help="Maximum number of sequences to evaluate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for sliding window (optional)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create benchmark
    benchmark = PerplexityBenchmark(device=device)
    
    # Run benchmark
    results = benchmark.run_benchmark(
        checkpoint_paths=[args.checkpoint],
        max_sequences=args.max_sequences,
        batch_size=args.batch_size,
        stride=args.stride
    )
    
    # Print and save results
    print(format_perplexity_results(results))
    save_perplexity_results(results)