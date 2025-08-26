#!/usr/bin/env python3
"""
Test script for benchmark functionality.
Run this to verify the benchmark system is working correctly.
"""

import sys
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from lm_eval import evaluator, tasks
    from lm_eval_wrapper import TransformerLMWrapper
    print("✓ lm-evaluation-harness imported successfully")
except ImportError as e:
    print(f"✗ Failed to import lm-evaluation-harness: {e}")
    print("  Please install with: pip install lm-eval")
    sys.exit(1)

# Test wrapper creation
print("\nTesting model wrapper...")
checkpoint_dir = Path("checkpoints")

if checkpoint_dir.exists():
    # Find a checkpoint to test
    checkpoints = list(checkpoint_dir.glob("**/*.pt"))
    
    if checkpoints:
        test_checkpoint = checkpoints[0]
        print(f"  Testing with checkpoint: {test_checkpoint}")
        
        try:
            wrapper = TransformerLMWrapper(
                checkpoint_path=str(test_checkpoint),
                device="cpu",  # Use CPU for testing
                batch_size=1
            )
            print("✓ Model wrapper created successfully")
            
            # Test tokenization
            test_text = "Hello world"
            tokens = wrapper.tok_encode(test_text)
            decoded = wrapper.tok_decode(tokens)
            print(f"✓ Tokenization test passed")
            print(f"  Original: '{test_text}'")
            print(f"  Tokens: {tokens[:10]}...")
            print(f"  Decoded: '{decoded}'")
            
            # Test simple evaluation
            print("\nTesting evaluation on small sample...")
            try:
                # Test loglikelihood
                result = wrapper.loglikelihood([("The cat", " sat")])
                print(f"✓ Loglikelihood test passed: {result[0][0]:.3f}")
                
                # Test generation
                generated = wrapper.generate_until([{"context": "Once upon a", "max_gen_toks": 5}])
                print(f"✓ Generation test passed: '{generated[0][:50]}...'")
                
                print("\n✅ All tests passed! Benchmark system is ready to use.")
                print("\nYou can now run Inference.py and select option 3 for benchmark evaluation.")
                
            except Exception as e:
                print(f"✗ Evaluation test failed: {e}")
                
        except Exception as e:
            print(f"✗ Failed to create wrapper: {e}")
            print("  This might be due to missing tokenizer or model issues")
    else:
        print("✗ No checkpoints found in checkpoints/")
        print("  Please train a model first")
else:
    print(f"✗ Checkpoint directory not found: {checkpoint_dir}")
    print("  Please train a model first")

print("\nAvailable benchmark tasks:")
try:
    # List some available tasks
    common_tasks = ["wikitext", "lambada_openai", "hellaswag", "piqa", "arc_easy", "winogrande"]
    available = []
    
    for task in common_tasks:
        try:
            # Check if task exists
            task_dict = tasks.get_task_dict([task])
            available.append(task)
        except:
            pass
    
    if available:
        print(f"✓ Found {len(available)} benchmark tasks:")
        for task in available:
            print(f"  • {task}")
    else:
        print("⚠ No benchmark tasks found. You may need to download them.")
        
except Exception as e:
    print(f"✗ Failed to check available tasks: {e}")