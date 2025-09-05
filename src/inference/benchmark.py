#!/usr/bin/env python3
"""Benchmark module for evaluating language models on the LAMBADA dataset.

LAMBADA (Language Modeling Broadened to Account for Discourse Aspects) tests 
context-dependent word prediction, requiring models to use long-range context 
to predict the final word in passages.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from tqdm import tqdm

# For downloading datasets
import urllib.request

# Import model loading utilities from Inference
from src.utils.model import TransformerLM
from src.dataset_preparation.tokenizer import CodeLlamaTokenizer


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
    


def _load_tokenizer() -> CodeLlamaTokenizer:
    """Load or download the CodeLlama tokenizer.
    
    Returns:
        Loaded CodeLlamaTokenizer instance
    """
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
        tokenizer.train()  # Downloads the pre-trained tokenizer
        tokenizer.save(str(tokenizer_path))
        print(f"CodeLlama tokenizer downloaded and saved to {tokenizer_path}")
    
    return tokenizer


def load_model_from_checkpoint(checkpoint_path: Union[str, Path], device: torch.device) -> Tuple[TransformerLM, CodeLlamaTokenizer]:
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
    ) -> Dict[str, Tuple[TransformerLM, CodeLlamaTokenizer]]:
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
        benchmark_config: Optional[Dict] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run LAMBADA benchmark on multiple models.
        
        Args:
            models_dict: Dictionary mapping model names to (model, tokenizer) tuples
            checkpoint_paths: Paths to checkpoint files to load models from
            benchmark_config: Optional configuration for benchmark (lambada_samples count)
            
        Returns:
            Dictionary of LAMBADA accuracy results for each model
        """
        # Validate inputs
        if models_dict is None and checkpoint_paths is None:
            raise ValueError("Either models_dict or checkpoint_paths must be provided")
        
        if models_dict is not None and checkpoint_paths is not None:
            raise ValueError("Cannot provide both models_dict and checkpoint_paths")
        
        # Load models from checkpoints if needed
        if checkpoint_paths is not None:
            models_dict = self.load_models_from_checkpoints(checkpoint_paths)
        
        # Default configuration
        config = benchmark_config or {
            'lambada_samples': 100  # Default to 100 samples for meaningful evaluation
        }
        
        # Load LAMBADA dataset
        print("\n" + "="*60)
        print("Loading LAMBADA Dataset...")
        print("="*60)
        
        lambada_samples = self.datasets.fetch_lambada_test(config['lambada_samples'])
        print(f"✓ Loaded {len(lambada_samples)} LAMBADA test samples")
        
        results = {}
        
        # Benchmark each model
        for model_name, (model, tokenizer) in models_dict.items():
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_name}")
            print(f"{'='*60}")
            
            model_results = {}
            
            try:
                # Evaluate LAMBADA Accuracy
                print("\nLAMBADA Dataset - Context-Dependent Last Word Prediction")
                print("   Evaluating strict accuracy...")
                lambada_acc = self.evaluate_lambada_accuracy(
                    model, tokenizer, lambada_samples
                )
                model_results['lambada_accuracy'] = lambada_acc
                model_results['num_samples'] = len(lambada_samples)
                print(f"   ✓ Accuracy: {lambada_acc:.2f}%")
                print(f"   ✓ Samples: {len(lambada_samples)}")
                
            except Exception as e:
                print(f"   ERROR: {str(e)}")
                model_results['error'] = str(e)
            
            results[model_name] = model_results
        
        return results


def format_benchmark_results(results: Dict[str, Dict[str, Any]]) -> str:
    """Format LAMBADA benchmark results as a readable table."""
    output = []
    output.append("\n" + "="*80)
    output.append("LAMBADA BENCHMARK RESULTS")
    output.append("="*80)
    output.append("\nLAMBADA (Language Modeling Broadened to Account for Discourse Aspects)")
    output.append("Tests context-dependent word prediction requiring long-range understanding")
    output.append("-"*80)
    
    # Find best score
    best_model = None
    best_score = -float('inf')
    for model_name, scores in results.items():
        if 'lambada_accuracy' in scores and scores['lambada_accuracy'] > best_score:
            best_score = scores['lambada_accuracy']
            best_model = model_name
    
    # Create results table
    output.append(f"\n{'Model':<40} | {'Accuracy':>12} | {'Samples':>10}")
    output.append("-"*80)
    
    # Add results for each model
    for model_name, scores in results.items():
        if 'error' in scores:
            output.append(f"{model_name[:40]:<40} | {'ERROR':>12} | {'-':>10}")
        elif 'lambada_accuracy' in scores:
            acc = scores['lambada_accuracy']
            num_samples = scores.get('num_samples', 'N/A')
            
            # Format accuracy with best marker
            acc_str = f"{acc:.2f}%"
            if model_name == best_model:
                acc_str += " *"
            
            output.append(f"{model_name[:40]:<40} | {acc_str:>12} | {num_samples:>10}")
    
    output.append("\n" + "="*80)
    output.append("INTERPRETATION:")
    output.append("  * indicates best performing model")
    output.append("  Higher accuracy indicates better context understanding")
    output.append("  Scores above 70% are considered strong performance")
    output.append("="*80)
    
    # Add summary if multiple models
    if len(results) > 1 and best_model:
        output.append("\n" + "="*80)
        output.append("SUMMARY:")
        output.append(f"  Best Model: {best_model}")
        output.append(f"  Best Accuracy: {best_score:.2f}%")
        
        # Calculate statistics
        accuracies = [s['lambada_accuracy'] for s in results.values() if 'lambada_accuracy' in s]
        if accuracies:
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            output.append(f"  Average Accuracy: {avg_acc:.2f}%")
            output.append(f"  Standard Deviation: {std_acc:.2f}%")
        output.append("="*80)
    
    return "\n".join(output)


def save_benchmark_results(results: Dict[str, Dict[str, Any]], output_dir: str = "benchmark_results"):
    """Save LAMBADA benchmark results to JSON with timestamp."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"lambada_benchmark_{timestamp}.json"
    
    # Add metadata
    full_results = {
        'benchmark': 'LAMBADA',
        'timestamp': timestamp,
        'results': results,
        'summary': {
            'num_models': len(results),
            'metric': 'lambada_accuracy',
            'description': 'Context-dependent last word prediction accuracy'
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    # Also save formatted text version
    text_filename = output_path / f"lambada_benchmark_{timestamp}.txt"
    with open(text_filename, 'w') as f:
        f.write(format_benchmark_results(results))
    
    print(f"Text summary saved to: {text_filename}")
