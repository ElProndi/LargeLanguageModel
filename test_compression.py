#!/usr/bin/env python3
"""Test script to analyze tokenizer compression on cleaned Wikipedia corpus."""

import sys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from tokenizer import WikipediaTokenizer


def load_sample_articles(file_path: str, num_articles: int = 1000) -> List[str]:
    """Load a sample of articles from a cleaned corpus file.
    
    Args:
        file_path: Path to cleaned corpus file
        num_articles: Number of articles to sample (default 1000)
        
    Returns:
        List of article texts
    """
    articles = []
    print(f"\nLoading {num_articles} articles from {Path(file_path).name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_articles:
                break
            line = line.strip()
            if line:  # Skip empty lines
                articles.append(line)
    
    print(f"Loaded {len(articles)} non-empty articles")
    return articles


def analyze_compression(tokenizer: WikipediaTokenizer, articles: List[str]) -> Dict:
    """Analyze compression metrics for a tokenizer on given articles using batch processing.
    
    Args:
        tokenizer: Trained WikipediaTokenizer instance
        articles: List of article texts to analyze
        
    Returns:
        Dictionary containing compression metrics
    """
    print(f"\nAnalyzing compression with vocab_size={tokenizer.vocab_size}...")
    print(f"  Processing {len(articles):,} articles in single batch...")
    
    # Count characters
    char_counts = [len(article) for article in articles]
    total_chars = sum(char_counts)
    
    # Tokenize ALL articles at once - the tokenizer will handle batch processing internally
    print(f"  Encoding all {len(articles):,} articles (this may take a moment)...")
    all_tokens = tokenizer.encode(articles, add_special_tokens=False)
    
    # Count tokens and calculate compression ratios
    token_counts = []
    compression_ratios = []
    total_tokens = 0
    
    for article_chars, article_tokens in zip(char_counts, all_tokens):
        token_count = len(article_tokens)
        total_tokens += token_count
        token_counts.append(token_count)
        
        # Calculate compression ratio for this article
        if token_count > 0:
            compression_ratio = article_chars / token_count
            compression_ratios.append(compression_ratio)
    
    print(f"  ✓ Processed {len(articles):,} articles")
    
    # Calculate statistics
    avg_compression = total_chars / total_tokens if total_tokens > 0 else 0
    
    # Calculate percentiles for compression ratios
    compression_ratios = np.array(compression_ratios)
    
    metrics = {
        "vocab_size": tokenizer.vocab_size,
        "num_articles": len(articles),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "avg_compression_ratio": avg_compression,
        "avg_chars_per_article": np.mean(char_counts),
        "avg_tokens_per_article": np.mean(token_counts),
        "compression_percentiles": {
            "p10": float(np.percentile(compression_ratios, 10)),
            "p25": float(np.percentile(compression_ratios, 25)),
            "p50": float(np.percentile(compression_ratios, 50)),
            "p75": float(np.percentile(compression_ratios, 75)),
            "p90": float(np.percentile(compression_ratios, 90)),
            "min": float(np.min(compression_ratios)),
            "max": float(np.max(compression_ratios))
        },
        "memory_savings_percent": (1 - total_tokens / total_chars) * 100
    }
    
    return metrics


def test_reconstruction(tokenizer: WikipediaTokenizer, test_texts: List[str], num_tests: int = 10):
    """Test that tokenization preserves the original text exactly.
    
    Args:
        tokenizer: Trained WikipediaTokenizer instance
        test_texts: List of texts to test
        num_tests: Number of texts to test (default 10)
        
    Returns:
        True if all tests pass, False otherwise
    """
    print(f"\nTesting reconstruction accuracy (sample of {num_tests} articles)...")
    
    all_passed = True
    for i, text in enumerate(test_texts[:num_tests]):
        # Encode and decode
        tokens = tokenizer.encode(text, add_special_tokens=False)
        reconstructed = tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Check if reconstruction matches original
        if text != reconstructed:
            print(f"  ✗ Article {i+1} failed reconstruction")
            print(f"    Original length: {len(text)}")
            print(f"    Reconstructed length: {len(reconstructed)}")
            # Show first difference
            for j, (c1, c2) in enumerate(zip(text, reconstructed)):
                if c1 != c2:
                    print(f"    First difference at position {j}: '{c1}' vs '{c2}'")
                    break
            all_passed = False
        else:
            print(f"  ✓ Article {i+1} passed (chars: {len(text)}, tokens: {len(tokens)})")
    
    return all_passed


def print_metrics_summary(metrics: Dict):
    """Pretty print compression metrics.
    
    Args:
        metrics: Dictionary of compression metrics
    """
    print("\n" + "="*60)
    print(f"COMPRESSION METRICS SUMMARY (vocab_size={metrics['vocab_size']})")
    print("="*60)
    
    print(f"\nDataset Statistics:")
    print(f"  Articles analyzed:        {metrics['num_articles']:,}")
    print(f"  Total characters:         {metrics['total_characters']:,}")
    print(f"  Total tokens:             {metrics['total_tokens']:,}")
    
    print(f"\nCompression Performance:")
    print(f"  Average compression ratio: {metrics['avg_compression_ratio']:.2f}x")
    print(f"  Memory savings:           {metrics['memory_savings_percent']:.1f}%")
    print(f"  Avg chars per article:    {metrics['avg_chars_per_article']:.1f}")
    print(f"  Avg tokens per article:   {metrics['avg_tokens_per_article']:.1f}")
    
    print(f"\nCompression Ratio Distribution:")
    percentiles = metrics['compression_percentiles']
    print(f"  Min:    {percentiles['min']:.2f}x")
    print(f"  P10:    {percentiles['p10']:.2f}x")
    print(f"  P25:    {percentiles['p25']:.2f}x")
    print(f"  Median: {percentiles['p50']:.2f}x")
    print(f"  P75:    {percentiles['p75']:.2f}x")
    print(f"  P90:    {percentiles['p90']:.2f}x")
    print(f"  Max:    {percentiles['max']:.2f}x")
    
    print("\n" + "="*60)


def analyze_full_tokenizer(articles: List[str]):
    """Analyze compression metrics for the full production tokenizer.
    
    Args:
        articles: List of article texts to analyze
    """
    print("\n" + "#"*60)
    print("FULL TOKENIZER COMPRESSION ANALYSIS")
    print("#"*60)
    
    # Load full tokenizer
    tokenizer_path = Path("tokenizers/full_tokenizer")
    
    if not tokenizer_path.exists():
        print(f"\nError: Full tokenizer not found at {tokenizer_path}")
        print("Please run 'python3 tokenizer.py' to train the full tokenizer first.")
        return None
    
    print(f"\nLoading full tokenizer from {tokenizer_path}...")
    tokenizer = WikipediaTokenizer()
    tokenizer.load(str(tokenizer_path))
    print(f"✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")
    
    # Test reconstruction accuracy
    print("\nValidating reconstruction accuracy...")
    reconstruction_passed = test_reconstruction(tokenizer, articles, num_tests=10)
    
    if reconstruction_passed:
        print("✓ Perfect reconstruction confirmed on all test samples")
    else:
        print("✗ Warning: Reconstruction errors detected - tokenizer may need retraining")
    
    # Analyze compression on full dataset
    metrics = analyze_compression(tokenizer, articles)
    
    # Print detailed summary
    print_metrics_summary(metrics)
    
    # Additional statistics for large-scale analysis
    if metrics['num_articles'] >= 10000:
        print(f"\nLarge-Scale Statistics:")
        print(f"  Total text processed:     {metrics['total_characters']/1024/1024:.2f} MB")
        print(f"  Tokenized size:           {metrics['total_tokens']*2/1024/1024:.2f} MB (uint16)")
        print(f"  Compression factor:       {metrics['avg_compression_ratio']:.3f}x")
        print(f"  Effective bits per char:  {16/metrics['avg_compression_ratio']:.2f} bits")
    
    return {"full_tokenizer": metrics}


def main():
    """Main test function."""
    print("="*60)
    print("TOKENIZER COMPRESSION TEST")
    print("="*60)
    
    # Configuration
    corpus_file = "/home/andrea/Desktop/data/cleaned_articles/cleaned_0.txt"
    num_articles = 100000  # Number of articles to analyze
    
    # Check if corpus file exists
    if not Path(corpus_file).exists():
        print(f"\nError: Corpus file not found: {corpus_file}")
        print("Please run cleaner.py first to generate cleaned articles.")
        sys.exit(1)
    
    # Load sample articles
    print(f"\nUsing corpus file: {corpus_file}")
    articles = load_sample_articles(corpus_file, num_articles)
    
    if not articles:
        print("\nError: No articles loaded from corpus file")
        sys.exit(1)
    
    # Show sample statistics
    sample_lengths = [len(a) for a in articles]
    print(f"\nSample article lengths:")
    print(f"  Min:    {min(sample_lengths):,} chars")
    print(f"  Median: {int(np.median(sample_lengths)):,} chars")
    print(f"  Mean:   {np.mean(sample_lengths):.1f} chars")
    print(f"  Max:    {max(sample_lengths):,} chars")
    
    # Analyze full tokenizer compression
    results = analyze_full_tokenizer(articles)
    
    if results:
        # Save results to JSON
        output_file = "compression_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo results to save - tokenizer analysis failed")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()