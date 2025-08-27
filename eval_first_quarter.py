#!/usr/bin/env python3
"""Evaluation script for checkpoint_first_fourth_complete.pt on first quarter of dataset."""

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Import project modules
from model import TransformerLM
from tokenizer import WikipediaTokenizer
from dataloader import create_simple_train_val_dataloaders


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal formatting."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_section(text: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}▶ {text}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * 50}{Colors.ENDC}")


def load_checkpoint_and_model():
    """Load the specific checkpoint and initialize model."""
    checkpoint_path = Path("checkpoints/run_20250827_132841/checkpoint_first_fourth_complete.pt")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print_section("Loading Model and Checkpoint")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {Colors.GREEN}{device}{Colors.ENDC}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        with open("config.json", 'r') as f:
            config = json.load(f)
    
    # Initialize model
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        max_position_embeddings=config['model']['max_position_embeddings'],
        dropout=config['model']['dropout'],
        attention_dropout=config['model']['attention_dropout'],
        layer_norm_eps=config['model']['layer_norm_eps'],
        initializer_range=config['model']['initializer_range'],
        use_cache=config['model'].get('use_cache', True),
        pad_token_id=config['tokenizer']['pad_token_id']
    )
    
    # Load state dict (handle compiled model format)
    state_dict = checkpoint['model_state_dict']
    # Remove "_orig_mod." prefix if present (from torch.compile)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            cleaned_state_dict[key[10:]] = value
        else:
            cleaned_state_dict[key] = value
    
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    # Print checkpoint info
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint step: {checkpoint.get('step', 'N/A')}")
    if 'train_loss' in checkpoint:
        print(f"Training loss at checkpoint: {checkpoint['train_loss']:.4f}")
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {Colors.CYAN}{total_params:,}{Colors.ENDC} "
          f"(trainable: {trainable_params:,})")
    
    # Load tokenizer
    print(f"\nLoading CodeLlama tokenizer...")
    tokenizer = WikipediaTokenizer()
    tokenizer.load("tokenizers/codellama_tokenizer")
    print(f"Vocabulary size: {Colors.CYAN}{tokenizer.vocab_size}{Colors.ENDC}")
    
    return model, tokenizer, device, checkpoint


def evaluate_on_validation(model, val_loader, device):
    """Evaluate model on validation set."""
    print_section("Evaluating on Validation Set")
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    num_batches = 0
    
    # Limit evaluation to 100 batches for speed
    max_batches = min(100, len(val_loader))
    
    print(f"Evaluating on {max_batches} batches...")
    
    start_time = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= max_batches:
                break
            
            # Move batch to device
            batch = batch.to(device, non_blocking=True)
            
            # Forward pass with labels = input_ids (shifted internally)
            outputs = model(batch, labels=batch)
            
            # Accumulate loss
            loss = outputs['loss'].item()
            batch_size = batch.shape[0]
            seq_len = batch.shape[1]
            
            total_loss += loss * batch_size
            total_tokens += batch_size * seq_len
            num_batches += 1
            
            # Progress indicator
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{max_batches} batches...")
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed
    
    print(f"\n{Colors.GREEN}Validation Results:{Colors.ENDC}")
    print(f"  Average Loss: {Colors.CYAN}{avg_loss:.4f}{Colors.ENDC}")
    print(f"  Perplexity: {Colors.CYAN}{perplexity:.2f}{Colors.ENDC}")
    print(f"  Evaluation Time: {elapsed:.2f}s")
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    
    return avg_loss, perplexity


def generate_samples(model, tokenizer, val_loader, device, num_samples=5):
    """Generate text samples from random validation sequences."""
    print_section("Generating Sample Predictions")
    
    model.eval()
    
    # Get random batch
    random_idx = random.randint(0, min(100, len(val_loader)) - 1)
    for idx, batch in enumerate(val_loader):
        if idx == random_idx:
            sample_batch = batch
            break
    
    # Move to device
    sample_batch = sample_batch.to(device)
    
    # Parameters for generation
    max_new_tokens = 50
    temperature = 0.8
    top_k = 50
    top_p = 0.95
    
    print(f"Generation parameters:")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Top-p: {top_p}")
    print(f"\nGenerating {num_samples} samples:\n")
    
    for i in range(min(num_samples, sample_batch.shape[0])):
        # Use first 50 tokens as prompt
        prompt_length = min(50, sample_batch.shape[1])
        prompt_ids = sample_batch[i, :prompt_length]
        
        # Decode prompt
        prompt_text = tokenizer.decode(prompt_ids.cpu().tolist())
        
        print(f"{Colors.BLUE}Sample {i+1}:{Colors.ENDC}")
        print(f"{Colors.WARNING}Prompt:{Colors.ENDC} {prompt_text[:100]}...")
        
        # Generate continuation
        start_time = time.time()
        
        # Prepare for generation
        input_ids = prompt_ids.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model predictions
                outputs = model(input_ids)
                logits = outputs['logits']
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Check for EOS token
                if next_token.item() == tokenizer.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_ids = input_ids[0, prompt_length:].cpu().tolist()
        generated_text = tokenizer.decode(generated_ids)
        
        elapsed = time.time() - start_time
        tokens_per_sec = len(generated_ids) / elapsed
        
        print(f"{Colors.GREEN}Generated:{Colors.ENDC} {generated_text}")
        print(f"{Colors.CYAN}Stats:{Colors.ENDC} {len(generated_ids)} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tokens/sec)")
        print("-" * 50)


def main():
    """Main evaluation function."""
    print_header("First Quarter Evaluation Script")
    
    # Load model and checkpoint
    model, tokenizer, device, checkpoint = load_checkpoint_and_model()
    
    # Load first quarter of dataset
    print_section("Loading First Quarter Dataset")
    print("Creating dataloaders with first quarter subset...")
    
    train_loader, val_loader, dataset_info = create_simple_train_val_dataloaders(
        batch_size=32,
        val_split=0.1,
        shuffle_train=False,  # No need to shuffle for evaluation
        test_mode=False,
        verbose=True,
        seed=42,
        file_subset='first_fourth'  # Load first quarter
    )
    
    print(f"\nDataset statistics:")
    print(f"  Train sequences: {Colors.CYAN}{dataset_info['train_size']:,}{Colors.ENDC}")
    print(f"  Val sequences: {Colors.CYAN}{dataset_info['val_size']:,}{Colors.ENDC}")
    # Window size is 512 as per config
    print(f"  Window size: {Colors.CYAN}512{Colors.ENDC} tokens")
    
    # Evaluate on validation set
    avg_loss, perplexity = evaluate_on_validation(model, val_loader, device)
    
    # Generate sample predictions
    generate_samples(model, tokenizer, val_loader, device, num_samples=10)
    
    # Print summary
    print_header("Evaluation Complete")
    print(f"Model: checkpoint_first_fourth_complete.pt")
    print(f"Dataset: First quarter (files 0-9 of 38)")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.2f}")
    
    # Memory cleanup
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()