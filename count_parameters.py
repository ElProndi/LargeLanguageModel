#!/usr/bin/env python3
"""
Script to count parameters for different model sizes.
Loads one model at a time and clears GPU memory between loads.
"""

import gc
import json
import torch
from src.utils.model import create_model


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """Format large numbers with commas and millions/billions notation."""
    if num >= 1e9:
        return f"{num:,} ({num/1e9:.2f}B)"
    elif num >= 1e6:
        return f"{num:,} ({num/1e6:.2f}M)"
    else:
        return f"{num:,}"


def main():
    # Load config to get available model sizes
    with open("config.json", "r") as f:
        config = json.load(f)
    
    model_sizes = list(config["models"].keys())
    
    print("=" * 70)
    print("Model Parameter Count Analysis")
    print("=" * 70)
    
    # Store results for final summary
    results = []
    
    for size in model_sizes:
        print(f"\n📊 Loading '{size}' model...")
        
        try:
            # Create model
            model = create_model("config.json", model_size=size)
            
            # Count parameters
            total_params, trainable_params = count_parameters(model)
            
            # Get model configuration details
            model_config = config["models"][size]
            
            # Store results
            results.append({
                "size": size,
                "total": total_params,
                "trainable": trainable_params,
                "config": model_config
            })
            
            # Print immediate results
            print(f"\n✅ Model: {size.upper()}")
            print(f"   Configuration:")
            print(f"     - Hidden size: {model_config['hidden_size']}")
            print(f"     - Layers: {model_config['num_layers']}")
            print(f"     - Attention heads: {model_config['num_heads']}")
            print(f"     - Vocab size: {model_config['vocab_size']}")
            print(f"   Parameters:")
            print(f"     - Total: {format_number(total_params)}")
            print(f"     - Trainable: {format_number(trainable_params)}")
            
            # Clear model and GPU memory
            print(f"   🧹 Clearing memory...")
            del model
            
            # Clear GPU cache if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            print(f"   ✓ Memory cleared")
            
        except Exception as e:
            print(f"   ❌ Error loading {size} model: {e}")
            continue
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model Size':<12} {'Hidden':<8} {'Layers':<8} {'Heads':<8} {'Total Parameters':<25} {'Trainable':<25}")
    print("-" * 70)
    
    for result in results:
        cfg = result["config"]
        print(f"{result['size'].upper():<12} "
              f"{cfg['hidden_size']:<8} "
              f"{cfg['num_layers']:<8} "
              f"{cfg['num_heads']:<8} "
              f"{format_number(result['total']):<25} "
              f"{format_number(result['trainable']):<25}")
    
    print("=" * 70)
    
    # Check final GPU memory status
    if torch.cuda.is_available():
        print(f"\n📊 Final GPU Memory Status:")
        print(f"   - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"   - Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()