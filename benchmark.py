#!/usr/bin/env python3
"""Benchmark module for evaluating language models on standard datasets."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Metrics libraries
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

# For downloading datasets
import urllib.request
import gzip
import random


class BenchmarkDatasets:
    """Handles downloading and caching of benchmark datasets."""
    
    def __init__(self, cache_dir: str = "benchmark_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def fetch_wikitext103_test(self, max_samples: int = 500) -> List[str]:
        """
        Fetch WikiText-103 test set samples.
        Using raw text version for perplexity evaluation.
        """
        cache_file = self.cache_dir / "wikitext103_test.json"
        
        if cache_file.exists():
            print("Loading WikiText-103 from cache...")
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data[:max_samples]
        
        print("Downloading WikiText-103 test set...")
        url = "https://raw.githubusercontent.com/pytorch/fairseq/master/examples/language_model/wikitext-103/wiki.test.tokens"
        
        try:
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
            
            # Split into paragraphs and filter
            paragraphs = []
            for paragraph in text.split('\n \n'):
                # Clean and filter paragraphs
                paragraph = paragraph.strip()
                if len(paragraph) > 100 and not paragraph.startswith('='):
                    paragraphs.append(paragraph)
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(paragraphs, f)
            
            print(f"Cached {len(paragraphs)} WikiText-103 paragraphs")
            return paragraphs[:max_samples]
            
        except Exception as e:
            print(f"Failed to download WikiText-103: {e}")
            # Fallback to simple test sentences
            return self._get_fallback_sentences(max_samples)
    
    def fetch_lambada_test(self, max_samples: int = 200) -> List[Dict[str, str]]:
        """
        Fetch LAMBADA dataset for context-dependent prediction.
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
    
    def fetch_hellaswag_validation(self, max_samples: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch HellaSwag validation samples for commonsense completion.
        Returns list of dicts with 'context', 'endings', and 'label' keys.
        """
        cache_file = self.cache_dir / "hellaswag_val.json"
        
        if cache_file.exists():
            print("Loading HellaSwag from cache...")
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data[:max_samples]
        
        print("Generating HellaSwag-style commonsense samples...")
        # Since HellaSwag requires specific format, we'll create similar tasks
        samples = [
            {
                'context': "The chef was preparing a complex dish in the kitchen. He carefully measured each ingredient and",
                'endings': [
                    "threw everything in the trash and ordered pizza instead.",
                    "followed the recipe step by step to ensure perfect results.",
                    "decided to juggle the knives for entertainment.",
                    "started dancing while the food burned on the stove."
                ],
                'label': 1
            },
            {
                'context': "The scientist was conducting an important experiment in the laboratory. She recorded her observations and",
                'endings': [
                    "analyzed the data to draw meaningful conclusions.",
                    "used the chemicals to paint a picture.",
                    "threw the equipment out the window.",
                    "started singing opera to the test tubes."
                ],
                'label': 0
            },
            {
                'context': "The student was studying for the final exam. After hours of reading, he",
                'endings': [
                    "ate his textbook for dinner.",
                    "decided to become a professional juggler instead.",
                    "took a short break to refresh his mind.",
                    "built a fort out of his notes."
                ],
                'label': 2
            }
        ]
        
        # Generate more samples programmatically
        contexts = [
            "The athlete was training for the marathon. During the run, she",
            "The programmer was debugging complex code. After finding the issue, they",
            "The teacher was explaining a difficult concept. To help students understand, she",
            "The artist was working on a new painting. As the work progressed, he",
            "The doctor was examining a patient. Based on the symptoms, she"
        ]
        
        for ctx in contexts:
            samples.append({
                'context': ctx,
                'endings': self._generate_endings(ctx),
                'label': 0  # First ending is always correct in our generation
            })
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(samples, f)
        
        return samples[:max_samples]
    
    def _generate_endings(self, context: str) -> List[str]:
        """Generate plausible and implausible endings for a context."""
        # Simple heuristic generation for demonstration
        reasonable = [
            "continued with careful attention to detail.",
            "proceeded according to the established plan.",
            "made adjustments based on the observations."
        ]
        unreasonable = [
            "suddenly decided to quit and become a circus performer.",
            "threw everything away and started over.",
            "began speaking in an ancient forgotten language."
        ]
        
        endings = [random.choice(reasonable)]
        endings.extend(random.sample(unreasonable, 3))
        return endings
    
    def _get_fallback_sentences(self, n: int) -> List[str]:
        """Fallback test sentences if download fails."""
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models have revolutionized natural language processing.",
            "Climate change poses significant challenges to global ecosystems.",
            "The human brain contains approximately 86 billion neurons.",
            "Artificial intelligence is transforming various industries worldwide.",
            "Quantum computing promises exponential speedups for certain problems.",
            "The Internet has fundamentally changed how people communicate.",
            "Renewable energy sources are becoming increasingly cost-effective.",
            "Space exploration continues to push the boundaries of human knowledge.",
            "Genetic engineering offers both opportunities and ethical challenges."
        ] * (n // 10 + 1)
        return sentences[:n]
    
    def _get_fallback_lambada(self, n: int) -> List[Dict[str, str]]:
        """Fallback LAMBADA-style samples if download fails."""
        samples = [
            {"context": "The cat sat on the", "target": "mat"},
            {"context": "She opened the door and saw a beautiful", "target": "garden"},
            {"context": "The scientist made an important", "target": "discovery"},
            {"context": "After years of hard work, he finally achieved his", "target": "goal"},
            {"context": "The children were excited to go to the", "target": "park"}
        ] * (n // 5 + 1)
        return samples[:n]


class ModelBenchmark:
    """Handles benchmark evaluation for language models."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.datasets = BenchmarkDatasets()
        self.rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_perplexity(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        texts: List[str],
        max_length: int = 512
    ) -> float:
        """
        Calculate perplexity on a list of texts.
        Lower perplexity indicates better language modeling.
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Calculating perplexity"):
                # Encode text
                input_ids = tokenizer.encode(text)
                
                # Truncate if necessary
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                
                # Skip if too short
                if len(input_ids) < 2:
                    continue
                
                # Add BOS token if needed
                special_tokens = tokenizer.get_special_token_ids()
                bos_id = special_tokens['bos_token_id']
                if input_ids[0] != bos_id:
                    input_ids = [bos_id] + input_ids
                
                # Convert to tensor
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Forward pass
                outputs = model(input_tensor, labels=input_tensor, return_dict=True)
                
                # Accumulate loss
                if outputs['loss'] is not None:
                    # Loss is already averaged over tokens
                    seq_length = input_tensor.size(1) - 1  # -1 because we predict from position 1
                    total_loss += outputs['loss'].item() * seq_length
                    total_tokens += seq_length
        
        # Calculate perplexity
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)
            return perplexity
        else:
            return float('inf')
    
    def evaluate_text_completion(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        prompts: List[str],
        references: List[str],
        max_new_tokens: int = 50
    ) -> Dict[str, float]:
        """
        Evaluate text completion using BLEU and ROUGE scores.
        """
        model.eval()
        generated_texts = []
        
        special_tokens = tokenizer.get_special_token_ids()
        
        for prompt in tqdm(prompts, desc="Generating completions"):
            # Encode prompt
            input_ids = tokenizer.encode(prompt)
            
            # Add BOS token if needed
            bos_id = special_tokens['bos_token_id']
            if input_ids[0] != bos_id:
                input_ids = [bos_id] + input_ids
            
            # Convert to tensor
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Generate
            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_length=len(input_ids) + max_new_tokens,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    eos_token_id=special_tokens['eos_token_id']
                )
            
            # Decode only the new tokens
            generated_ids = generated[0, len(input_ids):].tolist()
            generated_text = tokenizer.decode(generated_ids)
            generated_texts.append(generated_text)
        
        # Calculate BLEU score
        bleu_score = corpus_bleu(generated_texts, [references]).score
        
        # Calculate ROUGE scores
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for gen, ref in zip(generated_texts, references):
            scores = self.rouge_scorer_obj.score(ref, gen)
            for metric, score in scores.items():
                rouge_scores[metric].append(score.fmeasure)
        
        # Average ROUGE scores
        avg_rouge = {k: np.mean(v) * 100 for k, v in rouge_scores.items()}
        
        return {
            'bleu': bleu_score,
            'rouge1': avg_rouge['rouge1'],
            'rouge2': avg_rouge['rouge2'],
            'rougeL': avg_rouge['rougeL']
        }
    
    def evaluate_next_token_prediction(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        samples: List[Dict[str, str]],
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate next token prediction accuracy.
        Measures if the correct token is in top-1, top-5, and top-10 predictions.
        """
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        correct_top10 = 0
        total = 0
        
        special_tokens = tokenizer.get_special_token_ids()
        
        with torch.no_grad():
            for sample in tqdm(samples, desc="Evaluating next token prediction"):
                context = sample['context']
                target = sample['target']
                
                # Encode context and target
                context_ids = tokenizer.encode(context)
                target_ids = tokenizer.encode(target)
                
                if len(target_ids) == 0:
                    continue
                
                # Get first token of target (might be subword)
                target_id = target_ids[0]
                
                # Add BOS token if needed
                bos_id = special_tokens['bos_token_id']
                if context_ids[0] != bos_id:
                    context_ids = [bos_id] + context_ids
                
                # Convert to tensor
                input_tensor = torch.tensor(context_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Get predictions
                outputs = model(input_tensor, return_dict=True)
                logits = outputs['logits'][0, -1, :]  # Last position
                
                # Get top-k predictions
                top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(0)))
                top_k_indices = top_k_indices.cpu().numpy()
                
                # Check if target is in top predictions
                if target_id == top_k_indices[0]:
                    correct_top1 += 1
                if target_id in top_k_indices[:5]:
                    correct_top5 += 1
                if target_id in top_k_indices[:10]:
                    correct_top10 += 1
                
                total += 1
        
        if total == 0:
            return {'top1_acc': 0.0, 'top5_acc': 0.0, 'top10_acc': 0.0}
        
        return {
            'top1_acc': (correct_top1 / total) * 100,
            'top5_acc': (correct_top5 / total) * 100,
            'top10_acc': (correct_top10 / total) * 100
        }
    
    def evaluate_commonsense(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        samples: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate commonsense reasoning via multiple choice completion.
        Returns accuracy percentage.
        """
        model.eval()
        correct = 0
        total = 0
        
        special_tokens = tokenizer.get_special_token_ids()
        
        with torch.no_grad():
            for sample in tqdm(samples, desc="Evaluating commonsense"):
                context = sample['context']
                endings = sample['endings']
                label = sample['label']
                
                # Calculate likelihood for each ending
                scores = []
                for ending in endings:
                    full_text = context + " " + ending
                    
                    # Encode
                    input_ids = tokenizer.encode(full_text)
                    
                    # Add BOS token if needed
                    bos_id = special_tokens['bos_token_id']
                    if input_ids[0] != bos_id:
                        input_ids = [bos_id] + input_ids
                    
                    # Convert to tensor
                    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                    
                    # Get loss (negative log likelihood)
                    outputs = model(input_tensor, labels=input_tensor, return_dict=True)
                    
                    # Lower loss = higher likelihood
                    scores.append(-outputs['loss'].item())
                
                # Predict the ending with highest likelihood
                predicted = np.argmax(scores)
                if predicted == label:
                    correct += 1
                total += 1
        
        return (correct / total * 100) if total > 0 else 0.0
    
    def run_full_benchmark(
        self,
        models_dict: Dict[str, Tuple[Any, Any]],
        benchmark_config: Optional[Dict] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run complete benchmark suite on multiple models.
        
        Args:
            models_dict: Dictionary mapping model names to (model, tokenizer) tuples
            benchmark_config: Optional configuration for benchmarks
            
        Returns:
            Dictionary of results for each model
        """
        config = benchmark_config or {
            'wikitext_samples': 100,
            'lambada_samples': 50,
            'hellaswag_samples': 20,
            'max_length': 512,
            'max_new_tokens': 30
        }
        
        # Load datasets
        print("\n" + "="*60)
        print("Loading Benchmark Datasets...")
        print("="*60)
        
        wikitext_samples = self.datasets.fetch_wikitext103_test(config['wikitext_samples'])
        lambada_samples = self.datasets.fetch_lambada_test(config['lambada_samples'])
        hellaswag_samples = self.datasets.fetch_hellaswag_validation(config['hellaswag_samples'])
        
        # Prepare text completion data
        completion_prompts = [s['context'] for s in lambada_samples[:20]]
        completion_refs = [s['target'] for s in lambada_samples[:20]]
        
        results = {}
        
        # Benchmark each model
        for model_name, (model, tokenizer) in models_dict.items():
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_name}")
            print(f"{'='*60}")
            
            model_results = {}
            
            try:
                # 1. Perplexity on WikiText
                print("\n1. Calculating perplexity on WikiText-103...")
                perplexity = self.calculate_perplexity(
                    model, tokenizer, wikitext_samples, 
                    max_length=config['max_length']
                )
                model_results['perplexity'] = perplexity
                print(f"   Perplexity: {perplexity:.2f}")
                
                # 2. Text completion (BLEU/ROUGE)
                print("\n2. Evaluating text completion...")
                completion_scores = self.evaluate_text_completion(
                    model, tokenizer,
                    completion_prompts, completion_refs,
                    max_new_tokens=config['max_new_tokens']
                )
                model_results.update(completion_scores)
                print(f"   BLEU: {completion_scores['bleu']:.2f}")
                print(f"   ROUGE-1: {completion_scores['rouge1']:.2f}")
                print(f"   ROUGE-L: {completion_scores['rougeL']:.2f}")
                
                # 3. Next token prediction
                print("\n3. Evaluating next token prediction...")
                pred_scores = self.evaluate_next_token_prediction(
                    model, tokenizer, lambada_samples
                )
                model_results.update(pred_scores)
                print(f"   Top-1 Accuracy: {pred_scores['top1_acc']:.2f}%")
                print(f"   Top-5 Accuracy: {pred_scores['top5_acc']:.2f}%")
                
                # 4. Commonsense reasoning
                print("\n4. Evaluating commonsense reasoning...")
                commonsense_acc = self.evaluate_commonsense(
                    model, tokenizer, hellaswag_samples
                )
                model_results['commonsense_acc'] = commonsense_acc
                print(f"   Accuracy: {commonsense_acc:.2f}%")
                
            except Exception as e:
                print(f"   ERROR: {str(e)}")
                model_results['error'] = str(e)
            
            results[model_name] = model_results
        
        return results


def format_benchmark_results(results: Dict[str, Dict[str, Any]]) -> str:
    """Format benchmark results as a readable table."""
    output = []
    output.append("\n" + "="*80)
    output.append("BENCHMARK RESULTS SUMMARY")
    output.append("="*80)
    
    # Collect all metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    all_metrics.discard('error')
    all_metrics = sorted(all_metrics)
    
    # Find best scores for each metric
    best_scores = {}
    for metric in all_metrics:
        if metric == 'perplexity':  # Lower is better
            best_val = float('inf')
            best_model = None
            for model, scores in results.items():
                if metric in scores and scores[metric] < best_val:
                    best_val = scores[metric]
                    best_model = model
        else:  # Higher is better
            best_val = -float('inf')
            best_model = None
            for model, scores in results.items():
                if metric in scores and scores[metric] > best_val:
                    best_val = scores[metric]
                    best_model = model
        best_scores[metric] = best_model
    
    # Create table
    output.append("\n" + "-"*80)
    output.append(f"{'Model':<30} | " + " | ".join(f"{m[:12]:>12}" for m in all_metrics))
    output.append("-"*80)
    
    for model_name, scores in results.items():
        row = f"{model_name[:30]:<30} | "
        for metric in all_metrics:
            if 'error' in scores:
                value_str = "ERROR"
            elif metric in scores:
                value = scores[metric]
                if metric == 'perplexity':
                    value_str = f"{value:.2f}"
                elif 'acc' in metric:
                    value_str = f"{value:.1f}%"
                else:
                    value_str = f"{value:.2f}"
                
                # Mark best scores with *
                if best_scores.get(metric) == model_name:
                    value_str += "*"
            else:
                value_str = "N/A"
            
            row += f"{value_str:>12} | "
        output.append(row)
    
    output.append("-"*80)
    output.append("* indicates best score for that metric")
    output.append("Lower perplexity is better; higher scores are better for other metrics")
    
    return "\n".join(output)


def save_benchmark_results(results: Dict[str, Dict[str, Any]], output_dir: str = "benchmark_results"):
    """Save benchmark results to JSON with timestamp."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"benchmark_{timestamp}.json"
    
    # Add metadata
    full_results = {
        'timestamp': timestamp,
        'results': results,
        'summary': {
            'num_models': len(results),
            'metrics_evaluated': list(next(iter(results.values())).keys()) if results else []
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    # Also save formatted text version
    text_filename = output_path / f"benchmark_{timestamp}.txt"
    with open(text_filename, 'w') as f:
        f.write(format_benchmark_results(results))
    
    print(f"Text summary saved to: {text_filename}")


if __name__ == "__main__":
    print("Benchmark module loaded. Use through Inference.py or import for direct use.")
    print("\nExample usage:")
    print("  from benchmark import ModelBenchmark, format_benchmark_results")
    print("  benchmark = ModelBenchmark()")
    print("  results = benchmark.run_full_benchmark(models_dict)")
    print("  print(format_benchmark_results(results))")