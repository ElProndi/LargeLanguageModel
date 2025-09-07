#!/usr/bin/env python3
"""HellaSwag benchmark module for evaluating commonsense reasoning in language models.

HellaSwag (Harder Endings, Longer contexts, and Low-shot Activities for Situations 
With Adversarial Generations) tests a model's ability to predict plausible continuations
for real-world scenarios, requiring commonsense reasoning.
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
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# For downloading datasets
import urllib.request
import gzip

# Import model utilities
try:
    # Try relative imports (when imported as a module)
    from ..dataset_preparation.tokenizer import LLaMA2Tokenizer
    from ..utils.model import TransformerLM
except ImportError:
    # Fall back to absolute imports (when run directly)
    from src.dataset_preparation.tokenizer import LLaMA2Tokenizer
    from src.utils.model import TransformerLM


def _load_tokenizer() -> LLaMA2Tokenizer:
    """Load or download the LLaMA-2 tokenizer."""
    print("Loading tokenizer...")
    tokenizer = LLaMA2Tokenizer()
    tokenizer_path = Path("tokenizers/llama2_tokenizer")
    
    if tokenizer_path.exists():
        try:
            tokenizer.load(str(tokenizer_path))
            print(f"LLaMA-2 tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            raise
    else:
        print("CodeLlama tokenizer not found locally. Downloading from HuggingFace...")
        tokenizer.train()
        tokenizer.save(str(tokenizer_path))
        print(f"CodeLlama tokenizer downloaded and saved to {tokenizer_path}")
    
    return tokenizer


def load_model_from_checkpoint(checkpoint_path: Union[str, Path], device: torch.device) -> Tuple[TransformerLM, LLaMA2Tokenizer]:
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


class HellaSwagDataset:
    """Handles downloading and caching of the HellaSwag dataset."""
    
    def __init__(self, cache_dir: str = "benchmark_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def fetch_hellaswag(self, split: str = "validation", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch HellaSwag dataset from HuggingFace.
        
        Args:
            split: Dataset split to fetch ("train", "validation", "test")
            max_samples: Maximum number of samples to return
            
        Returns:
            List of HellaSwag samples with context, endings, and label
        """
        cache_file = self.cache_dir / f"hellaswag_{split}.json"
        
        if cache_file.exists():
            print(f"Loading HellaSwag {split} from cache...")
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if max_samples:
                    return data[:max_samples]
                return data
        
        print(f"Downloading HellaSwag {split} set...")
        
        # HuggingFace dataset URL for HellaSwag
        urls = {
            "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
            "validation": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
            "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl"
        }
        
        if split not in urls:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(urls.keys())}")
        
        url = urls[split]
        
        try:
            samples = []
            with urllib.request.urlopen(url) as response:
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line:
                        sample = json.loads(line)
                        # Extract relevant fields
                        processed_sample = {
                            'ctx': sample.get('ctx', ''),  # Context
                            'endings': sample.get('endings', []),  # 4 possible endings
                            'label': sample.get('label', -1),  # Correct answer (0-3)
                            'ctx_a': sample.get('ctx_a', ''),  # Additional context A
                            'ctx_b': sample.get('ctx_b', ''),  # Additional context B  
                            'activity_label': sample.get('activity_label', ''),  # Activity type
                            'source_id': sample.get('source_id', '')  # Source identifier
                        }
                        
                        # Only include samples with valid labels for evaluation
                        if processed_sample['label'] >= 0:
                            samples.append(processed_sample)
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(samples, f)
            
            print(f"Cached {len(samples)} HellaSwag {split} samples")
            
            if max_samples:
                return samples[:max_samples]
            return samples
            
        except Exception as e:
            print(f"Failed to download HellaSwag: {e}")
            return self._get_fallback_hellaswag(max_samples or 5)
    
    def _get_fallback_hellaswag(self, n: int) -> List[Dict[str, Any]]:
        """Fallback HellaSwag samples if download fails."""
        samples = [
            {
                "ctx": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She",
                "endings": [
                    "rinses the bucket off with soap and blow dries the dog's head.",
                    "uses a hose to rinse him off and he runs away again.",
                    "gets the dog into the bath but the dog jumps out immediately.",
                    "decides to give up and takes the dog for a walk instead."
                ],
                "label": 1,
                "activity_label": "Bathing dog"
            },
            {
                "ctx": "A man is standing in a kitchen. He picks up a knife and",
                "endings": [
                    "starts chopping vegetables on the cutting board.",
                    "throws it at the wall in frustration.",
                    "begins juggling with three knives.",
                    "puts it in the dishwasher."
                ],
                "label": 0,
                "activity_label": "Cooking"
            }
        ] * (n // 2 + 1)
        return samples[:n]


class HellaSwagBenchmark:
    """Handles HellaSwag commonsense reasoning evaluation for language models."""
    
    def __init__(self, device: torch.device = None, verbose: bool = True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.dataset = HellaSwagDataset()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better tokenization."""
        # Clean up text
        text = text.strip()
        # Ensure space after punctuation for better tokenization
        text = text.replace('.', '. ').replace(',', ', ').replace('!', '! ').replace('?', '? ')
        # Remove multiple spaces
        while '  ' in text:
            text = text.replace('  ', ' ')
        return text.strip()
    
    def calculate_sequence_logprob(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        context: str,
        completion: str,
        max_length: int = 512
    ) -> float:
        """
        Calculate log probability of a completion given a context.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for text encoding
            context: The context text
            completion: The completion text to score
            max_length: Maximum sequence length
            
        Returns:
            Average log probability per token for the completion
        """
        # Combine context and completion
        full_text = context + " " + completion
        
        # Tokenize
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        context_tokens = tokenizer.encode(context, add_special_tokens=True)
        
        # Truncate if necessary
        if len(tokens) > max_length:
            # Keep the end of the sequence (most relevant for completion scoring)
            tokens = tokens[-max_length:]
            # Adjust context length accordingly
            context_len = len(context_tokens) - (len(tokens) - max_length)
            context_len = max(1, context_len)  # Ensure at least 1 context token
        else:
            context_len = len(context_tokens)
        
        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)
            logits = outputs['logits']
        
        # Calculate log probabilities for the completion tokens
        # We need to align: logits[i] predicts token[i+1]
        shift_logits = logits[0, context_len-1:-1, :]  # Logits that predict completion tokens
        shift_labels = input_ids[0, context_len:]  # Actual completion tokens
        
        # Calculate log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Average log probability
        avg_log_prob = token_log_probs.mean().item()
        
        return avg_log_prob
    
    def evaluate_hellaswag(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        samples: List[Dict[str, Any]],
        batch_process: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on HellaSwag samples.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for text encoding
            samples: List of HellaSwag samples
            batch_process: Whether to process in batches (currently processes one at a time)
            
        Returns:
            Dictionary with accuracy metrics
        """
        model.eval()
        correct = 0
        total = 0
        predictions = []
        
        for sample in tqdm(samples, desc="Evaluating HellaSwag"):
            context = self.preprocess_text(sample['ctx'])
            endings = [self.preprocess_text(e) for e in sample['endings']]
            label = sample['label']
            
            # Skip invalid samples
            if label < 0 or label >= len(endings):
                continue
            
            # Calculate log probability for each ending
            scores = []
            for ending in endings:
                score = self.calculate_sequence_logprob(
                    model, tokenizer, context, ending
                )
                scores.append(score)
            
            # Predict the ending with highest log probability
            predicted = np.argmax(scores)
            predictions.append({
                'predicted': int(predicted),
                'label': label,
                'scores': scores,
                'correct': predicted == label
            })
            
            if predicted == label:
                correct += 1
            total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        # Calculate additional metrics
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions
        }
        
        return results
    
    def run_benchmark(
        self,
        checkpoint_paths: Union[List[Union[str, Path]], Dict[str, Union[str, Path]]],
        max_samples: Optional[int] = 500,
        split: str = "validation"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run HellaSwag benchmark on multiple models.
        
        Args:
            checkpoint_paths: Paths to model checkpoints
            max_samples: Maximum samples to evaluate
            split: Dataset split to use
            
        Returns:
            Dictionary mapping model names to results
        """
        # Normalize input to dict format
        if isinstance(checkpoint_paths, list):
            checkpoint_dict = {Path(p).stem: p for p in checkpoint_paths}
        else:
            checkpoint_dict = checkpoint_paths
        
        # Load dataset
        print("\n" + "="*60)
        print("Loading HellaSwag Dataset...")
        print("="*60)
        
        samples = self.dataset.fetch_hellaswag(split=split, max_samples=max_samples)
        print(f"Loaded {len(samples)} HellaSwag {split} samples")
        
        results = {}
        
        print("\n" + "="*60)
        print("HELLASWAG BENCHMARK - Commonsense Reasoning")
        print("="*60)
        
        for name, checkpoint_path in checkpoint_dict.items():
            print(f"\nEvaluating: {name}")
            print("-"*40)
            
            try:
                # Load model
                model, tokenizer = load_model_from_checkpoint(checkpoint_path, self.device)
                
                # Evaluate
                if self.verbose:
                    print(f"Running HellaSwag evaluation on {len(samples)} samples...")
                
                model_results = self.evaluate_hellaswag(
                    model=model,
                    tokenizer=tokenizer,
                    samples=samples
                )
                
                if self.verbose:
                    print(f"\nResults:")
                    print(f"  Accuracy: {model_results['accuracy']:.2f}%")
                    print(f"  Correct: {model_results['correct']}/{model_results['total']}")
                
                # Remove detailed predictions for summary (keep them for detailed analysis)
                summary_results = {k: v for k, v in model_results.items() if k != 'predictions'}
                results[name] = summary_results
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results[name] = {'error': str(e)}
            
            # Clean up
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
        
        return results


def format_hellaswag_results(results: Dict[str, Dict[str, Any]]) -> str:
    """Format HellaSwag benchmark results as a readable table."""
    output = []
    output.append("\n" + "="*80)
    output.append("HELLASWAG BENCHMARK RESULTS - Commonsense Reasoning")
    output.append("="*80)
    output.append("\nHellaSwag tests ability to predict plausible scenario continuations.")
    output.append("Human performance: ~95% | Random chance: 25%")
    output.append("-"*80)
    
    # Find best accuracy
    best_model = None
    best_accuracy = -float('inf')
    for model_name, scores in results.items():
        if 'accuracy' in scores and scores['accuracy'] > best_accuracy:
            best_accuracy = scores['accuracy']
            best_model = model_name
    
    # Create results table
    output.append(f"\n{'Model':<30} | {'Accuracy':>12} | {'Correct':>12} | {'Total':>10}")
    output.append("-"*80)
    
    # Add results for each model
    for model_name, scores in results.items():
        if 'error' in scores:
            output.append(f"{model_name[:30]:<30} | {'ERROR':>12} | {'-':>12} | {'-':>10}")
        elif 'accuracy' in scores:
            acc = scores['accuracy']
            correct = scores['correct']
            total = scores['total']
            
            # Format accuracy with best marker
            acc_str = f"{acc:.2f}%"
            if model_name == best_model:
                acc_str += " *"
            
            output.append(f"{model_name[:30]:<30} | {acc_str:>12} | {correct:>12} | {total:>10}")
    
    output.append("\n" + "="*80)
    output.append("INTERPRETATION:")
    output.append("  * indicates best performing model")
    output.append("  > 80%: Strong commonsense reasoning")
    output.append("  60-80%: Good performance")
    output.append("  40-60%: Moderate performance")
    output.append("  < 40%: Needs improvement")
    output.append("="*80)
    
    # Add summary if multiple models
    if len(results) > 1 and best_model:
        output.append("\n" + "="*80)
        output.append("SUMMARY:")
        output.append(f"  Best Model: {best_model}")
        output.append(f"  Best Accuracy: {best_accuracy:.2f}%")
        
        # Calculate statistics
        accuracies = [s['accuracy'] for s in results.values() if 'accuracy' in s]
        if accuracies:
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            output.append(f"  Average Accuracy: {avg_acc:.2f}%")
            output.append(f"  Standard Deviation: {std_acc:.2f}%")
        output.append("="*80)
    
    return "\n".join(output)


def save_hellaswag_results(results: Dict[str, Dict[str, Any]], output_dir: str = "benchmark_results"):
    """Save HellaSwag benchmark results to JSON with timestamp."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"hellaswag_benchmark_{timestamp}.json"
    
    # Add metadata
    full_results = {
        'benchmark': 'HellaSwag',
        'timestamp': timestamp,
        'results': results,
        'summary': {
            'num_models': len(results),
            'metric': 'accuracy',
            'description': 'Commonsense reasoning - predicting plausible scenario continuations'
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    # Also save formatted text version
    text_filename = output_path / f"hellaswag_benchmark_{timestamp}.txt"
    with open(text_filename, 'w') as f:
        f.write(format_hellaswag_results(results))
    
    print(f"Text summary saved to: {text_filename}")


# Main execution for standalone testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HellaSwag commonsense reasoning benchmark")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create benchmark
    benchmark = HellaSwagBenchmark(device=device)
    
    # Run benchmark
    results = benchmark.run_benchmark(
        checkpoint_paths=[args.checkpoint],
        max_samples=args.max_samples,
        split=args.split
    )
    
    # Print and save results
    print(format_hellaswag_results(results))
    save_hellaswag_results(results)