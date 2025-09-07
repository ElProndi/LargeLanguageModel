#!/usr/bin/env python3
"""Comprehensive benchmark module for evaluating language models.

This module provides a unified interface for running multiple benchmarks:
- LAMBADA: Context-dependent word prediction
- Perplexity: FineWeb held-out data evaluation  
- HellaSwag: Commonsense reasoning
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
from tqdm import tqdm

# For downloading datasets
import urllib.request

# Import model loading utilities from Inference
try:
    # Try relative imports (when imported as a module)
    from ..utils.model import TransformerLM
    from ..dataset_preparation.tokenizer import LLaMA2Tokenizer
    # Import new benchmark modules
    from .perplexity_benchmark import PerplexityBenchmark
    from .hellaswag_benchmark import HellaSwagBenchmark
except ImportError:
    # Fall back to absolute imports (when run directly)
    from src.utils.model import TransformerLM
    from src.dataset_preparation.tokenizer import LLaMA2Tokenizer
    # Import new benchmark modules
    from perplexity_benchmark import PerplexityBenchmark
    from hellaswag_benchmark import HellaSwagBenchmark


class BenchmarkDatasets:
    """Handles downloading and caching of the LAMBADA dataset."""
    
    def __init__(self, cache_dir: str = "benchmark_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def fetch_lambada_test(self, max_samples: int = 200) -> List[Dict[str, str]]:
        """
        Fetch LAMBADA dataset for last-word prediction accuracy.
        Returns list of dicts with 'context' and 'target' keys.
        """
        cache_file = self.cache_dir / "lambada_test.json"
        
        if cache_file.exists():
            print("Loading LAMBADA from cache...")
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data[:max_samples]
        
        print("Downloading LAMBADA test set...")
        url = "https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl"
        
        try:
            samples = []
            with urllib.request.urlopen(url) as response:
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line:
                        sample = json.loads(line)
                        # LAMBADA format: last word is the target
                        text = sample.get('text', '')
                        if text:
                            words = text.split()
                            if len(words) > 1:
                                context = ' '.join(words[:-1])
                                target = words[-1]
                                samples.append({'context': context, 'target': target})
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(samples, f)
            
            print(f"Cached {len(samples)} LAMBADA samples")
            return samples[:max_samples]
            
        except Exception as e:
            print(f"Failed to download LAMBADA: {e}")
            return self._get_fallback_lambada(max_samples)
    
    
    def _get_fallback_lambada(self, n: int) -> List[Dict[str, str]]:
        """Fallback LAMBADA samples if download fails."""
        samples = [
            {"context": "The cat sat on the", "target": "mat"},
            {"context": "She opened the door and saw a beautiful", "target": "garden"},
            {"context": "The scientist made an important", "target": "discovery"},
            {"context": "After years of hard work, he finally achieved his", "target": "goal"},
            {"context": "The children were excited to go to the", "target": "park"}
        ] * (n // 5 + 1)
        return samples[:n]
    


def _load_tokenizer() -> LLaMA2Tokenizer:
    """Load or download the CodeLlama tokenizer.
    
    Returns:
        Loaded LLaMA2Tokenizer instance
    """
    print("Loading tokenizer...")
    tokenizer = LLaMA2Tokenizer()
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
        tokenizer.train()  # Downloads the pre-trained tokenizer
        tokenizer.save(str(tokenizer_path))
        print(f"CodeLlama tokenizer downloaded and saved to {tokenizer_path}")
    
    return tokenizer


def load_model_from_checkpoint(checkpoint_path: Union[str, Path], device: torch.device) -> Tuple[TransformerLM, LLaMA2Tokenizer]:
    """Load model from checkpoint with proper architecture handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Note: weights_only=False is safe here since we're loading our own trained checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint, fallback to file if not present
    config = checkpoint.get('config')
    if not config:
        with open("config.json", 'r') as f:
            config = json.load(f)
    
    # Extract model configuration
    if 'models' in config:
        model_size = checkpoint.get('model_size')
        if not model_size:
            available_sizes = list(config['models'].keys())
            model_size = available_sizes[0]  # Use first available size
        model_config = config['models'][model_size]
    else:
        model_config = config['model']
    
    tokenizer_config = config.get('tokenizer', {})
    rope_config = config.get('rope', {})
    architecture_features = config.get('architecture_features', {})
    
    # Extract actual intermediate_size from the checkpoint's state_dict
    state_dict = checkpoint['model_state_dict']
    intermediate_size = None
    
    # Check for FFN layer dimensions to determine actual intermediate_size
    for key in state_dict.keys():
        if 'ffn.w_fused.weight' in key:
            # w_fused contains both gate and up projections
            intermediate_size = state_dict[key].shape[0] // 2
            break
        elif 'ffn.w_gate.weight' in key:
            # Old format: shape is [intermediate_size, hidden_size]
            intermediate_size = state_dict[key].shape[0]
            break
    
    # Log the settings being used
    print(f"  Flash Attention: ✓ Always Enabled")
    print(f"  Compilation: ✗ Disabled (benchmark mode)")
    
    # Create model with all architecture parameters from config
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        # Pass detected or configured intermediate_size
        intermediate_size=intermediate_size,  # Use detected size from checkpoint
        max_position_embeddings=model_config['max_position_embeddings'],
        # RoPE configuration
        rope_theta=rope_config.get('theta', 10000.0),
        rope_scaling=rope_config.get('scaling_factor', 1.0),
        # Architecture features from config - CRITICAL for correct model reconstruction
        tie_embeddings=architecture_features.get('tie_embeddings', True),
        # Disable dropout for inference
        dropout=0.0,
        attention_dropout=0.0,
        layer_norm_eps=model_config['layer_norm_eps'],
        initializer_range=model_config['initializer_range'],
        pad_token_id=tokenizer_config.get('pad_token_id', 2)  # Same as EOS token (CodeLlama convention)
    )
    
    # Handle compiled model state dict (remove "_orig_mod." prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v  # Remove "_orig_mod." prefix
        else:
            new_state_dict[k] = v
    
    # Load state dict and validate
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            raise RuntimeError(f"Model architecture doesn't match checkpoint: {e}")
        else:
            raise e
    
    model = model.to(device)
    model.eval()
    
    # Skip torch.compile() in benchmark mode to avoid Flash Attention issues
    # Benchmarking measures model quality, not inference speed
    
    # Get model stats
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")
    
    # Load tokenizer (shared across all models)
    tokenizer = _load_tokenizer()
    
    return model, tokenizer


class ModelBenchmark:
    """Handles LAMBADA benchmark evaluation for language models.
    
    The LAMBADA dataset evaluates a model's ability to predict the final word
    in passages where understanding the broader context is essential for
    accurate prediction.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.datasets = BenchmarkDatasets()
    
    def evaluate_lambada_accuracy(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        samples: List[Dict[str, str]]
    ) -> float:
        """
        Evaluate LAMBADA last-word prediction accuracy.
        Strict accuracy metric - exact match required.
        """
        model.eval()
        correct = 0
        total = 0
        
        special_tokens = tokenizer.get_special_token_ids()
        
        with torch.no_grad():
            for sample in tqdm(samples, desc="Evaluating LAMBADA accuracy"):
                context = sample['context']
                target = sample['target']
                
                # Encode context without special tokens; we will add BOS manually
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                
                # Add BOS token if needed
                bos_id = special_tokens['bos_token_id']
                if len(context_ids) == 0 or context_ids[0] != bos_id:
                    context_ids = [bos_id] + context_ids
                
                # Convert to tensor
                input_tensor = torch.tensor(context_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Get predictions
                outputs = model(input_tensor, return_dict=True)
                logits = outputs['logits'][0, -1, :]  # Last position
                
                # Get top prediction
                predicted_id = torch.argmax(logits).item()
                predicted_token = tokenizer.decode([predicted_id]).strip()
                
                # Check if prediction matches target (case-insensitive)
                if predicted_token.lower() == target.lower():
                    correct += 1
                
                total += 1
        
        return (correct / total * 100) if total > 0 else 0.0
    
    
    def load_models_from_checkpoints(
        self,
        checkpoint_paths: Union[List[Union[str, Path]], Dict[str, Union[str, Path]]]
    ) -> Dict[str, Tuple[TransformerLM, LLaMA2Tokenizer]]:
        """
        Load multiple models from checkpoint files.
        
        Args:
            checkpoint_paths: Either a list of checkpoint paths or a dict mapping names to paths
            
        Returns:
            Dictionary mapping model names to (model, tokenizer) tuples
        """
        models_dict = {}
        
        # Normalize input to dict format
        if isinstance(checkpoint_paths, list):
            # Create names from checkpoint filenames
            checkpoint_dict = {Path(p).stem: p for p in checkpoint_paths}
        else:
            checkpoint_dict = checkpoint_paths
        
        print(f"\nLoading {len(checkpoint_dict)} models from checkpoints...")
        print("=" * 60)
        
        for name, checkpoint_path in checkpoint_dict.items():
            checkpoint_path = Path(checkpoint_path)
            print(f"\nLoading: {name}")
            print(f"  From: {checkpoint_path}")
            
            try:
                model, tokenizer = load_model_from_checkpoint(checkpoint_path, self.device)
                models_dict[name] = (model, tokenizer)
                print(f"  ✓ Successfully loaded {name}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
                print(f"  Skipping this model...")
        
        print("\n" + "=" * 60)
        print(f"Successfully loaded {len(models_dict)}/{len(checkpoint_dict)} models")
        
        if not models_dict:
            raise ValueError("No models could be loaded successfully")
        
        return models_dict
    
    def run_full_benchmark(
        self,
        models_dict: Optional[Dict[str, Tuple[Any, Any]]] = None,
        checkpoint_paths: Optional[Union[List[Union[str, Path]], Dict[str, Union[str, Path]]]] = None,
        benchmark_config: Optional[Dict] = None,
        benchmarks_to_run: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run comprehensive benchmark suite on multiple models.
        
        Args:
            models_dict: Dictionary mapping model names to (model, tokenizer) tuples
            checkpoint_paths: Paths to checkpoint files to load models from
            benchmark_config: Optional configuration for benchmarks
            benchmarks_to_run: List of benchmarks to run. If None, runs all.
                              Options: ['lambada', 'perplexity', 'hellaswag']
            
        Returns:
            Dictionary of benchmark results for each model
        """
        # Validate inputs
        if models_dict is None and checkpoint_paths is None:
            raise ValueError("Either models_dict or checkpoint_paths must be provided")
        
        if models_dict is not None and checkpoint_paths is not None:
            raise ValueError("Cannot provide both models_dict and checkpoint_paths")
        
        # Default benchmarks to run
        if benchmarks_to_run is None:
            benchmarks_to_run = ['lambada', 'perplexity', 'hellaswag']
        
        # Default configuration
        config = benchmark_config or {
            'lambada_samples': 100,
            'perplexity_sequences': 500,
            'hellaswag_samples': 500
        }
        
        # Initialize benchmarks
        perplexity_bench = PerplexityBenchmark(device=self.device, verbose=False) if 'perplexity' in benchmarks_to_run else None
        hellaswag_bench = HellaSwagBenchmark(device=self.device, verbose=False) if 'hellaswag' in benchmarks_to_run else None
        
        # Load LAMBADA dataset if needed
        lambada_samples = None
        if 'lambada' in benchmarks_to_run:
            print("\n" + "="*60)
            print("Loading LAMBADA Dataset...")
            print("="*60)
            lambada_samples = self.datasets.fetch_lambada_test(config['lambada_samples'])
            print(f"✓ Loaded {len(lambada_samples)} LAMBADA test samples")
        
        # Load HellaSwag dataset if needed
        hellaswag_samples = None
        if 'hellaswag' in benchmarks_to_run and hellaswag_bench:
            print("\n" + "="*60)
            print("Loading HellaSwag Dataset...")
            print("="*60)
            hellaswag_samples = hellaswag_bench.dataset.fetch_hellaswag(
                split="validation",
                max_samples=config.get('hellaswag_samples', 500)
            )
            print(f"✓ Loaded {len(hellaswag_samples)} HellaSwag validation samples")
        
        # Main results dictionary
        results = {}
        
        # Process checkpoints directly if provided
        if checkpoint_paths is not None:
            # Normalize to dict
            if isinstance(checkpoint_paths, list):
                checkpoint_dict = {Path(p).stem: p for p in checkpoint_paths}
            else:
                checkpoint_dict = checkpoint_paths
            
            for model_name, checkpoint_path in checkpoint_dict.items():
                print(f"\n{'='*60}")
                print(f"Benchmarking: {model_name}")
                print(f"{'='*60}")
                
                model_results = {}
                
                try:
                    # Load model once for all benchmarks
                    model, tokenizer = load_model_from_checkpoint(checkpoint_path, self.device)
                    
                    # Run LAMBADA
                    if 'lambada' in benchmarks_to_run and lambada_samples:
                        print("\nLAMBADA - Context-Dependent Last Word Prediction")
                        lambada_acc = self.evaluate_lambada_accuracy(
                            model, tokenizer, lambada_samples
                        )
                        model_results['lambada_accuracy'] = lambada_acc
                        model_results['lambada_samples'] = len(lambada_samples)
                        print(f"   ✓ Accuracy: {lambada_acc:.2f}%")
                    
                    # Run Perplexity
                    if 'perplexity' in benchmarks_to_run and perplexity_bench:
                        print("\nPerplexity - FineWeb Held-Out Data")
                        ppl_results = perplexity_bench.evaluate_model(
                            model=model,
                            tokenizer=tokenizer,
                            max_sequences=config.get('perplexity_sequences', 500),
                            batch_size=32
                        )
                        model_results['perplexity'] = ppl_results['perplexity']
                        model_results['perplexity_loss'] = ppl_results['avg_loss']
                        print(f"   ✓ Perplexity: {ppl_results['perplexity']:.2f}")
                    
                    # Run HellaSwag
                    if 'hellaswag' in benchmarks_to_run and hellaswag_bench and hellaswag_samples:
                        print("\nHellaSwag - Commonsense Reasoning")
                        hs_results = hellaswag_bench.evaluate_hellaswag(
                            model=model,
                            tokenizer=tokenizer,
                            samples=hellaswag_samples
                        )
                        model_results['hellaswag_accuracy'] = hs_results['accuracy']
                        model_results['hellaswag_correct'] = hs_results['correct']
                        model_results['hellaswag_total'] = hs_results['total']
                        print(f"   ✓ Accuracy: {hs_results['accuracy']:.2f}%")
                    
                    # Clean up model
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"   ERROR: {str(e)}")
                    model_results['error'] = str(e)
                
                results[model_name] = model_results
        
        else:
            # Use pre-loaded models
            for model_name, (model, tokenizer) in models_dict.items():
                print(f"\n{'='*60}")
                print(f"Benchmarking: {model_name}")
                print(f"{'='*60}")
                
                model_results = {}
                
                try:
                    # Run LAMBADA
                    if 'lambada' in benchmarks_to_run and lambada_samples:
                        print("\nLAMBADA - Context-Dependent Last Word Prediction")
                        lambada_acc = self.evaluate_lambada_accuracy(
                            model, tokenizer, lambada_samples
                        )
                        model_results['lambada_accuracy'] = lambada_acc
                        model_results['lambada_samples'] = len(lambada_samples)
                        print(f"   ✓ Accuracy: {lambada_acc:.2f}%")
                    
                    # Run Perplexity
                    if 'perplexity' in benchmarks_to_run and perplexity_bench:
                        print("\nPerplexity - FineWeb Held-Out Data")
                        ppl_results = perplexity_bench.evaluate_model(
                            model=model,
                            tokenizer=tokenizer,
                            max_sequences=config.get('perplexity_sequences', 500),
                            batch_size=32
                        )
                        model_results['perplexity'] = ppl_results['perplexity']
                        model_results['perplexity_loss'] = ppl_results['avg_loss']
                        print(f"   ✓ Perplexity: {ppl_results['perplexity']:.2f}")
                    
                    # Run HellaSwag
                    if 'hellaswag' in benchmarks_to_run and hellaswag_bench and hellaswag_samples:
                        print("\nHellaSwag - Commonsense Reasoning")
                        hs_results = hellaswag_bench.evaluate_hellaswag(
                            model=model,
                            tokenizer=tokenizer,
                            samples=hellaswag_samples
                        )
                        model_results['hellaswag_accuracy'] = hs_results['accuracy']
                        model_results['hellaswag_correct'] = hs_results['correct']
                        model_results['hellaswag_total'] = hs_results['total']
                        print(f"   ✓ Accuracy: {hs_results['accuracy']:.2f}%")
                    
                except Exception as e:
                    print(f"   ERROR: {str(e)}")
                    model_results['error'] = str(e)
                
                results[model_name] = model_results
        
        return results


def format_benchmark_results(results: Dict[str, Dict[str, Any]]) -> str:
    """Format comprehensive benchmark results as a readable table."""
    output = []
    output.append("\n" + "="*100)
    output.append("COMPREHENSIVE BENCHMARK RESULTS")
    output.append("="*100)
    
    # Determine which benchmarks were run
    has_lambada = any('lambada_accuracy' in r for r in results.values())
    has_perplexity = any('perplexity' in r for r in results.values())
    has_hellaswag = any('hellaswag_accuracy' in r for r in results.values())
    
    # Create header based on available benchmarks
    header_parts = ["Model"]
    if has_lambada:
        header_parts.append("LAMBADA")
    if has_perplexity:
        header_parts.append("Perplexity")
    if has_hellaswag:
        header_parts.append("HellaSwag")
    
    # Build format string
    if has_lambada and has_perplexity and has_hellaswag:
        output.append(f"\n{'Model':<25} | {'LAMBADA':>12} | {'Perplexity':>12} | {'HellaSwag':>12}")
        output.append("-"*100)
    elif has_lambada and has_perplexity:
        output.append(f"\n{'Model':<35} | {'LAMBADA':>12} | {'Perplexity':>12}")
        output.append("-"*80)
    elif has_lambada and has_hellaswag:
        output.append(f"\n{'Model':<35} | {'LAMBADA':>12} | {'HellaSwag':>12}")
        output.append("-"*80)
    elif has_perplexity and has_hellaswag:
        output.append(f"\n{'Model':<35} | {'Perplexity':>12} | {'HellaSwag':>12}")
        output.append("-"*80)
    else:
        # Single benchmark
        output.append(f"\n{'Model':<40} | {'Score':>12}")
        output.append("-"*60)
    
    # Track best performers
    best_lambada = (None, -float('inf'))
    best_perplexity = (None, float('inf'))  # Lower is better
    best_hellaswag = (None, -float('inf'))
    
    # Add results for each model
    for model_name, scores in results.items():
        if 'error' in scores:
            if has_lambada and has_perplexity and has_hellaswag:
                output.append(f"{model_name[:25]:<25} | {'ERROR':>12} | {'ERROR':>12} | {'ERROR':>12}")
            elif has_lambada and has_perplexity:
                output.append(f"{model_name[:35]:<35} | {'ERROR':>12} | {'ERROR':>12}")
            elif has_lambada and has_hellaswag:
                output.append(f"{model_name[:35]:<35} | {'ERROR':>12} | {'ERROR':>12}")
            elif has_perplexity and has_hellaswag:
                output.append(f"{model_name[:35]:<35} | {'ERROR':>12} | {'ERROR':>12}")
            else:
                output.append(f"{model_name[:40]:<40} | {'ERROR':>12}")
        else:
            # Format individual scores
            lambada_str = f"{scores['lambada_accuracy']:.2f}%" if 'lambada_accuracy' in scores else "-"
            perplexity_str = f"{scores['perplexity']:.2f}" if 'perplexity' in scores else "-"
            hellaswag_str = f"{scores['hellaswag_accuracy']:.2f}%" if 'hellaswag_accuracy' in scores else "-"
            
            # Track best scores
            if 'lambada_accuracy' in scores and scores['lambada_accuracy'] > best_lambada[1]:
                best_lambada = (model_name, scores['lambada_accuracy'])
            if 'perplexity' in scores and scores['perplexity'] < best_perplexity[1]:
                best_perplexity = (model_name, scores['perplexity'])
            if 'hellaswag_accuracy' in scores and scores['hellaswag_accuracy'] > best_hellaswag[1]:
                best_hellaswag = (model_name, scores['hellaswag_accuracy'])
            
            # Output row
            if has_lambada and has_perplexity and has_hellaswag:
                output.append(f"{model_name[:25]:<25} | {lambada_str:>12} | {perplexity_str:>12} | {hellaswag_str:>12}")
            elif has_lambada and has_perplexity:
                output.append(f"{model_name[:35]:<35} | {lambada_str:>12} | {perplexity_str:>12}")
            elif has_lambada and has_hellaswag:
                output.append(f"{model_name[:35]:<35} | {lambada_str:>12} | {hellaswag_str:>12}")
            elif has_perplexity and has_hellaswag:
                output.append(f"{model_name[:35]:<35} | {perplexity_str:>12} | {hellaswag_str:>12}")
            elif has_lambada:
                output.append(f"{model_name[:40]:<40} | {lambada_str:>12}")
            elif has_perplexity:
                output.append(f"{model_name[:40]:<40} | {perplexity_str:>12}")
            elif has_hellaswag:
                output.append(f"{model_name[:40]:<40} | {hellaswag_str:>12}")
    
    # Add interpretation
    output.append("\n" + "="*100)
    output.append("INTERPRETATION:")
    if has_lambada:
        output.append("  LAMBADA: Context-dependent word prediction (higher is better, >70% is strong)")
    if has_perplexity:
        output.append("  Perplexity: How well model predicts text (lower is better, <50 is good)")
    if has_hellaswag:
        output.append("  HellaSwag: Commonsense reasoning (higher is better, >60% is good)")
    output.append("="*100)
    
    # Add best performers summary
    if len(results) > 1:
        output.append("\n" + "="*100)
        output.append("BEST PERFORMERS:")
        if best_lambada[0]:
            output.append(f"  LAMBADA:    {best_lambada[0]} ({best_lambada[1]:.2f}%)")
        if best_perplexity[0] and best_perplexity[1] != float('inf'):
            output.append(f"  Perplexity: {best_perplexity[0]} ({best_perplexity[1]:.2f})")
        if best_hellaswag[0]:
            output.append(f"  HellaSwag:  {best_hellaswag[0]} ({best_hellaswag[1]:.2f}%)")
        output.append("="*100)
    
    return "\n".join(output)


def save_benchmark_results(results: Dict[str, Dict[str, Any]], output_dir: str = "benchmark_results"):
    """Save comprehensive benchmark results to JSON with timestamp."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine which benchmarks were run
    has_lambada = any('lambada_accuracy' in r for r in results.values())
    has_perplexity = any('perplexity' in r for r in results.values())
    has_hellaswag = any('hellaswag_accuracy' in r for r in results.values())
    
    # Create appropriate filename
    if has_lambada and has_perplexity and has_hellaswag:
        filename = output_path / f"full_benchmark_{timestamp}.json"
        benchmark_name = "Full Benchmark Suite"
    elif has_lambada and has_perplexity:
        filename = output_path / f"lambada_perplexity_benchmark_{timestamp}.json"
        benchmark_name = "LAMBADA + Perplexity"
    elif has_lambada and has_hellaswag:
        filename = output_path / f"lambada_hellaswag_benchmark_{timestamp}.json"
        benchmark_name = "LAMBADA + HellaSwag"
    elif has_perplexity and has_hellaswag:
        filename = output_path / f"perplexity_hellaswag_benchmark_{timestamp}.json"
        benchmark_name = "Perplexity + HellaSwag"
    elif has_lambada:
        filename = output_path / f"lambada_benchmark_{timestamp}.json"
        benchmark_name = "LAMBADA"
    elif has_perplexity:
        filename = output_path / f"perplexity_benchmark_{timestamp}.json"
        benchmark_name = "Perplexity"
    elif has_hellaswag:
        filename = output_path / f"hellaswag_benchmark_{timestamp}.json"
        benchmark_name = "HellaSwag"
    else:
        filename = output_path / f"benchmark_{timestamp}.json"
        benchmark_name = "Benchmark"
    
    # Build metrics list
    metrics = []
    if has_lambada:
        metrics.append('lambada_accuracy')
    if has_perplexity:
        metrics.append('perplexity')
    if has_hellaswag:
        metrics.append('hellaswag_accuracy')
    
    # Add metadata
    full_results = {
        'benchmark': benchmark_name,
        'timestamp': timestamp,
        'results': results,
        'summary': {
            'num_models': len(results),
            'metrics': metrics,
            'description': 'Comprehensive language model evaluation'
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    # Also save formatted text version
    text_filename = filename.with_suffix('.txt')
    with open(text_filename, 'w') as f:
        f.write(format_benchmark_results(results))
    
    print(f"Text summary saved to: {text_filename}")
