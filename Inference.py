#!/usr/bin/env python3
"""Interactive inference script for transformer language model."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
import numpy as np

# Import project modules
from model import TransformerLM
from tokenizer import WikipediaTokenizer
from benchmark import ModelBenchmark, format_benchmark_results, save_benchmark_results


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
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_info(text: str):
    """Print info message in cyan."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message in yellow."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message in red."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def list_run_directories(base_checkpoint_dir: Path) -> List[Tuple[str, Path]]:
    """List all available training run directories.
    
    Returns:
        List of (display_name, path) tuples sorted by modification time
    """
    run_dirs = []
    
    # Get all subdirectories
    for run_dir in sorted(base_checkpoint_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if run_dir.is_dir() and any(run_dir.glob("*.pt")):
            # Get directory stats
            stats = run_dir.stat()
            modified = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            num_checkpoints = len(list(run_dir.glob("*.pt")))
            
            # Create display name
            display_name = f"{run_dir.name:<25} ({num_checkpoints:2d} checkpoints, {modified})"
            run_dirs.append((display_name, run_dir))
    
    return run_dirs


def list_checkpoints(checkpoint_dir: Path) -> List[Tuple[str, Path]]:
    """List all available checkpoints with details.
    
    Returns:
        List of (display_name, path) tuples
    """
    checkpoints = []
    
    # Get all .pt files
    for checkpoint_path in sorted(checkpoint_dir.glob("*.pt")):
        # Get file stats
        stats = checkpoint_path.stat()
        size = format_file_size(stats.st_size)
        modified = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create display name with details
        display_name = f"{checkpoint_path.name:<30} ({size:>10}, {modified})"
        checkpoints.append((display_name, checkpoint_path))
    
    return checkpoints


def select_run_directory(base_checkpoint_dir: Path) -> Optional[Path]:
    """Interactive run directory selection.
    
    Returns:
        Selected run directory path or None if cancelled
    """
    run_dirs = list_run_directories(base_checkpoint_dir)
    
    if not run_dirs:
        # No subdirectories, check if there are checkpoints in the base directory (old format)
        if any(base_checkpoint_dir.glob("*.pt")):
            print_warning("Found checkpoints in old flat structure. Using base directory.")
            return base_checkpoint_dir
        print_error(f"No training runs found in {base_checkpoint_dir}")
        return None
    
    print(f"{Colors.BOLD}Available Training Runs:{Colors.ENDC}")
    print("-" * 70)
    
    for idx, (display_name, _) in enumerate(run_dirs, 1):
        if idx == 1:  # Highlight most recent
            print(f"  {Colors.GREEN}{idx:2d}. {display_name} (most recent){Colors.ENDC}")
        else:
            print(f"  {idx:2d}. {display_name}")
    
    print("-" * 70)
    print(f"  {Colors.WARNING}0. Cancel{Colors.ENDC}")
    
    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Select training run (0-{len(run_dirs)}): {Colors.ENDC}")
            choice_idx = int(choice)
            
            if choice_idx == 0:
                return None
            elif 1 <= choice_idx <= len(run_dirs):
                _, selected_path = run_dirs[choice_idx - 1]
                print_success(f"Selected run: {selected_path.name}")
                return selected_path
            else:
                print_warning(f"Please enter a number between 0 and {len(run_dirs)}")
        except (ValueError, KeyboardInterrupt):
            print_warning("Invalid input. Please enter a number.")


def select_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Interactive checkpoint selection.
    
    Returns:
        Selected checkpoint path or None if cancelled
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print_error(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    print(f"{Colors.BOLD}Available Checkpoints:{Colors.ENDC}")
    print("-" * 70)
    
    for idx, (display_name, _) in enumerate(checkpoints, 1):
        # Highlight special checkpoints
        if "best" in display_name:
            print(f"  {Colors.GREEN}{idx:2d}. {display_name}{Colors.ENDC}")
        elif "latest" in display_name:
            print(f"  {Colors.BLUE}{idx:2d}. {display_name}{Colors.ENDC}")
        else:
            print(f"  {idx:2d}. {display_name}")
    
    print("-" * 70)
    print(f"  {Colors.WARNING}0. Cancel{Colors.ENDC}")
    
    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Select checkpoint (0-{len(checkpoints)}): {Colors.ENDC}")
            choice_idx = int(choice)
            
            if choice_idx == 0:
                return None
            elif 1 <= choice_idx <= len(checkpoints):
                _, selected_path = checkpoints[choice_idx - 1]
                print_success(f"Selected: {selected_path.name}")
                return selected_path
            else:
                print_warning(f"Please enter a number between 0 and {len(checkpoints)}")
        except (ValueError, KeyboardInterrupt):
            print_warning("Invalid input. Please enter a number.")


def select_multiple_checkpoints(checkpoint_dir: Path) -> Optional[List[Path]]:
    """Interactive multi-checkpoint selection.
    
    Returns:
        List of selected checkpoint paths or None if cancelled
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print_error(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    print(f"{Colors.BOLD}Available Checkpoints for Multi-Model Comparison:{Colors.ENDC}")
    print("-" * 70)
    
    # Display all checkpoints with indices
    for idx, (display_name, _) in enumerate(checkpoints, 1):
        if "best" in display_name:
            print(f"  {idx:2d}. {display_name} {Colors.GREEN}(best){Colors.ENDC}")
        elif "latest" in display_name:
            print(f"  {idx:2d}. {display_name} {Colors.BLUE}(latest){Colors.ENDC}")
        else:
            print(f"  {idx:2d}. {display_name}")
    
    print("-" * 70)
    print(f"{Colors.CYAN}Enter checkpoint numbers separated by commas (e.g., 1,3,5){Colors.ENDC}")
    print(f"{Colors.CYAN}Or enter 'all' to select all checkpoints{Colors.ENDC}")
    print(f"{Colors.WARNING}Enter 0 to cancel{Colors.ENDC}")
    
    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Select checkpoints: {Colors.ENDC}").strip()
            
            if choice == '0':
                return None
            
            if choice.lower() == 'all':
                selected_paths = [path for _, path in checkpoints]
                print_success(f"Selected all {len(selected_paths)} checkpoints")
                return selected_paths
            
            # Parse comma-separated numbers
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                selected_paths = []
                selected_names = []
                
                for idx in indices:
                    if 1 <= idx <= len(checkpoints):
                        name, path = checkpoints[idx - 1]
                        selected_paths.append(path)
                        selected_names.append(path.name)
                    else:
                        print_warning(f"Invalid index {idx}, skipping...")
                
                if selected_paths:
                    print_success(f"Selected {len(selected_paths)} checkpoints:")
                    for name in selected_names:
                        print(f"  • {name}")
                    return selected_paths
                else:
                    print_warning("No valid checkpoints selected. Please try again.")
                    
            except ValueError:
                print_warning("Invalid input. Please enter numbers separated by commas, 'all', or 0 to cancel.")
                
        except KeyboardInterrupt:
            print_warning("\nSelection cancelled")
            return None


def load_model_and_tokenizer(checkpoint_path: Path, device: torch.device) -> Tuple[TransformerLM, WikipediaTokenizer]:
    """Load model from checkpoint and tokenizer.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load checkpoint first to get the config
    print_info(f"Loading checkpoint: {checkpoint_path.name}...")
    # Note: weights_only=False is safe here since we're loading our own trained checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint, fallback to file if not present (for old checkpoints)
    if 'config' in checkpoint:
        print_info("Loading configuration from checkpoint...")
        config = checkpoint['config']
    else:
        print_warning("Config not found in checkpoint, loading from config.json")
        print_warning("This checkpoint may be from an older version")
        with open("config.json", 'r') as f:
            config = json.load(f)
    
    # Create model using config from checkpoint
    print_info("Creating model architecture from checkpoint config...")
    
    # Extract model config
    model_config = config['model']
    tokenizer_config = config.get('tokenizer', {})
    rope_config = config.get('rope', {})
    
    # Create model with all architecture parameters from config
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        # CRITICAL: Pass num_kv_heads to match trained model architecture
        num_kv_heads=model_config.get('num_kv_heads', model_config['num_heads']),
        # Pass modern architecture flags
        use_rms_norm=model_config.get('use_rms_norm', True),
        use_swiglu=model_config.get('use_swiglu', True),
        max_position_embeddings=model_config['max_position_embeddings'],
        # RoPE configuration
        rope_theta=rope_config.get('theta', 10000.0),
        rope_scaling=rope_config.get('scaling_factor', 1.0),
        force_flash_attention=model_config.get('force_flash_attention', True),
        # Disable dropout for inference
        dropout=0.0,
        attention_dropout=0.0,
        layer_norm_eps=model_config['layer_norm_eps'],
        initializer_range=model_config['initializer_range'],
        use_cache=model_config['use_cache'],
        pad_token_id=tokenizer_config.get('pad_token_id', 2)
    )
    
    # Handle compiled model state dict (remove "_orig_mod." prefix if present)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v  # Remove "_orig_mod." prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    # Get model stats
    total_params = sum(p.numel() for p in model.parameters())
    print_success(f"Model loaded: {total_params:,} parameters")
    
    # Display checkpoint info
    if 'global_step' in checkpoint:
        print_info(f"Training step: {checkpoint['global_step']:,}")
    if 'best_val_loss' in checkpoint:
        print_info(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Load tokenizer
    print_info("Loading tokenizer...")
    tokenizer = WikipediaTokenizer()
    
    # Load CodeLlama tokenizer
    tokenizer_path = Path("tokenizers/codellama_tokenizer")
    
    if tokenizer_path.exists():
        try:
            tokenizer.load(str(tokenizer_path))
            print_success(f"CodeLlama tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            print_error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            sys.exit(1)
    else:
        print_info("CodeLlama tokenizer not found locally. Downloading from HuggingFace...")
        tokenizer.train()  # Downloads the pre-trained tokenizer
        tokenizer.save(str(tokenizer_path))
        print_success(f"CodeLlama tokenizer downloaded and saved to {tokenizer_path}")
    
    return model, tokenizer


def load_multiple_models(
    checkpoint_paths: List[Path], 
    device: torch.device
) -> Tuple[Dict[str, Tuple[TransformerLM, WikipediaTokenizer]], float]:
    """Load multiple models from checkpoints.
    
    Returns:
        Tuple of (models_dict, total_memory_gb)
        where models_dict maps checkpoint names to (model, tokenizer) tuples
    """
    models = {}
    total_memory = 0
    
    # Check available memory first
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        available_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        free_memory = available_memory - used_memory
        print_info(f"Available GPU memory: {free_memory / 1e9:.2f} GB")
    
    print(f"\n{Colors.BOLD}Loading {len(checkpoint_paths)} Models...{Colors.ENDC}")
    print("-" * 60)
    
    for idx, checkpoint_path in enumerate(checkpoint_paths, 1):
        model_name = checkpoint_path.name
        print(f"\n{Colors.CYAN}Model {idx}/{len(checkpoint_paths)}: {model_name}{Colors.ENDC}")
        
        try:
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
            
            # Calculate memory usage
            model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            total_memory += model_memory
            
            # Store in dictionary
            models[model_name] = (model, tokenizer)
            
            print_success(f"✓ Loaded {model_name} ({model_memory / 1e6:.1f} MB)")
            
            # Check memory after each load
            if device.type == 'cuda':
                current_usage = torch.cuda.memory_allocated() / 1e9
                print_info(f"Current GPU memory usage: {current_usage:.2f} GB")
                
        except Exception as e:
            print_error(f"Failed to load {model_name}: {e}")
            print_warning(f"Continuing with other models...")
    
    print("\n" + "=" * 60)
    print_success(f"Successfully loaded {len(models)} out of {len(checkpoint_paths)} models")
    print_info(f"Total model memory: {total_memory / 1e9:.2f} GB")
    
    if not models:
        print_error("No models could be loaded!")
        return None, 0
    
    return models, total_memory / 1e9




def select_prompt_mode() -> Tuple[str, Optional[str]]:
    """Select prompt input mode.
    
    Returns:
        Tuple of (mode, prompt) where mode is 'custom' or 'exit'
    """
    print(f"\n{Colors.BOLD}Enter Prompt:{Colors.ENDC}")
    print("  Type your prompt or 'q' to quit")
    
    prompt = input(f"\n{Colors.BOLD}> {Colors.ENDC}")
    
    if prompt.lower() == 'q' or prompt.lower() == 'quit':
        return 'exit', None
    elif prompt.strip():
        return 'custom', prompt
    else:
        print_warning("Empty prompt. Please try again.")
        return select_prompt_mode()


def get_generation_params() -> Dict[str, any]:
    """Get generation parameters from user or use defaults.
    
    Returns:
        Dictionary of generation parameters
    """
    print(f"\n{Colors.BOLD}Generation Parameters:{Colors.ENDC}")
    print("  Press Enter to use default values")
    
    params = {}
    
    # Max length
    default_max_length = 200
    max_length_input = input(f"  Max length (default {default_max_length}): ")
    params['max_length'] = int(max_length_input) if max_length_input else default_max_length
    
    # Temperature
    default_temp = 0.8
    temp_input = input(f"  Temperature (default {default_temp}): ")
    params['temperature'] = float(temp_input) if temp_input else default_temp
    
    # Top-k
    default_top_k = 50
    top_k_input = input(f"  Top-k (default {default_top_k}, 0 to disable): ")
    params['top_k'] = int(top_k_input) if top_k_input else default_top_k
    params['top_k'] = params['top_k'] if params['top_k'] > 0 else None
    
    # Top-p
    default_top_p = 0.95
    top_p_input = input(f"  Top-p (default {default_top_p}, 1.0 to disable): ")
    params['top_p'] = float(top_p_input) if top_p_input else default_top_p
    params['top_p'] = params['top_p'] if params['top_p'] < 1.0 else None
    
    return params


def generate_text(
    model: TransformerLM,
    tokenizer: WikipediaTokenizer,
    prompt: str,
    device: torch.device,
    **generation_params
) -> str:
    """Generate text from prompt.
    
    Returns:
        Generated text string
    """
    # Encode prompt (no cleaning needed - CodeLlama handles Unicode)
    print_info(f"Encoding prompt...")
    input_ids = tokenizer.encode(prompt)
    
    # Add BOS token at the beginning if not present
    special_tokens = tokenizer.get_special_token_ids()
    bos_id = special_tokens['bos_token_id']
    if input_ids[0] != bos_id:
        input_ids = [bos_id] + input_ids
    
    # Convert to tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    print_info(f"Input tokens: {len(input_ids)}")
    
    # Generate
    print_info("Generating...")
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_tensor,
            eos_token_id=special_tokens['eos_token_id'],
            **generation_params
        )
    
    generation_time = time.time() - start_time
    
    # Decode generated text
    generated_ids = generated_ids[0].tolist()  # Get first batch element
    generated_text = tokenizer.decode(generated_ids)
    
    # Calculate statistics
    tokens_generated = len(generated_ids) - len(input_ids)
    tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
    
    # Print statistics
    print_success(f"Generation complete!")
    print_info(f"Tokens generated: {tokens_generated}")
    print_info(f"Time taken: {generation_time:.2f}s")
    print_info(f"Speed: {tokens_per_sec:.1f} tokens/sec")
    
    return generated_text


def generate_text_multi_model(
    models_dict: Dict[str, Tuple[TransformerLM, WikipediaTokenizer]],
    prompt: str,
    device: torch.device,
    **generation_params
) -> Dict[str, Dict[str, any]]:
    """Generate text from multiple models for comparison.
    
    Returns:
        Dictionary mapping model names to generation results
    """
    results = {}
    
    print(f"\n{Colors.BOLD}Generating from {len(models_dict)} models...{Colors.ENDC}")
    print(f"Prompt: {Colors.CYAN}{prompt}{Colors.ENDC}")
    print("-" * 60)
    
    for idx, (model_name, (model, tokenizer)) in enumerate(models_dict.items(), 1):
        print(f"\n{Colors.BLUE}[{idx}/{len(models_dict)}] Model: {model_name}{Colors.ENDC}")
        
        try:
            # Encode prompt  
            input_ids = tokenizer.encode(prompt)
            
            # Get special token IDs
            special_tokens = tokenizer.get_special_token_ids()
            bos_id = special_tokens['bos_token_id']
            
            # Add BOS token if needed
            if input_ids[0] != bos_id:
                input_ids = [bos_id] + input_ids
            
            # Convert to tensor
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate
            start_time = time.time()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_tensor,
                    eos_token_id=special_tokens['eos_token_id'],
                    **generation_params
                )
            
            generation_time = time.time() - start_time
            
            # Decode generated text
            generated_ids = generated_ids[0].tolist()
            generated_text = tokenizer.decode(generated_ids)
            
            # Calculate statistics
            tokens_generated = len(generated_ids) - len(input_ids)
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
            
            # Store results
            results[model_name] = {
                'text': generated_text,
                'tokens_generated': tokens_generated,
                'generation_time': generation_time,
                'tokens_per_sec': tokens_per_sec,
                'input_tokens': len(input_ids)
            }
            
            print_success(f"✓ Generated {tokens_generated} tokens in {generation_time:.2f}s")
            
        except Exception as e:
            print_error(f"Generation failed for {model_name}: {e}")
            results[model_name] = {
                'text': f"[ERROR: {str(e)}]",
                'tokens_generated': 0,
                'generation_time': 0,
                'tokens_per_sec': 0,
                'input_tokens': 0,
                'error': True
            }
    
    return results


def display_multi_model_results(results: Dict[str, Dict[str, any]], prompt: str):
    """Display comparison results from multiple models."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'MODEL COMPARISON RESULTS'.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Prompt:{Colors.ENDC} {Colors.CYAN}{prompt}{Colors.ENDC}\n")
    
    # Display each model's output
    for idx, (model_name, result) in enumerate(results.items(), 1):
        print(f"\n{Colors.BLUE}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BLUE}{Colors.BOLD}Model {idx}: {model_name}{Colors.ENDC}")
        print(f"{Colors.BLUE}{'─' * 70}{Colors.ENDC}")
        
        if result.get('error'):
            print(f"{Colors.RED}Generation failed: {result['text']}{Colors.ENDC}")
        else:
            print(f"\n{result['text']}\n")
            
            # Display statistics
            print(f"{Colors.CYAN}Statistics:{Colors.ENDC}")
            print(f"  • Tokens generated: {result['tokens_generated']}")
            print(f"  • Generation time: {result['generation_time']:.2f}s")
            print(f"  • Speed: {result['tokens_per_sec']:.1f} tokens/sec")
    
    # Summary comparison
    print(f"\n{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
    print(f"{Colors.GREEN}{Colors.BOLD}Performance Summary:{Colors.ENDC}")
    print(f"{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
    
    valid_results = {k: v for k, v in results.items() if not v.get('error')}
    if valid_results:
        # Find fastest and most productive
        fastest = min(valid_results.items(), key=lambda x: x[1]['generation_time'])
        most_tokens = max(valid_results.items(), key=lambda x: x[1]['tokens_generated'])
        highest_speed = max(valid_results.items(), key=lambda x: x[1]['tokens_per_sec'])
        
        print(f"  • Fastest generation: {fastest[0]} ({fastest[1]['generation_time']:.2f}s)")
        print(f"  • Most tokens: {most_tokens[0]} ({most_tokens[1]['tokens_generated']} tokens)")
        print(f"  • Highest speed: {highest_speed[0]} ({highest_speed[1]['tokens_per_sec']:.1f} tok/s)")
    
    print(f"{Colors.GREEN}{'=' * 70}{Colors.ENDC}")


def main():
    """Main inference loop."""
    print_header("Transformer Language Model Inference")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print_success(f"Using GPU: {torch.cuda.get_device_name()}")
        print_info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print_warning("CUDA not available. Using CPU (will be slower)")
    
    # Select inference mode
    print(f"\n{Colors.BOLD}Select Inference Mode:{Colors.ENDC}")
    print("  1. Single model inference")
    print("  2. Multi-model comparison")
    print("  3. Benchmark models")
    print("  0. Exit")
    
    while True:
        try:
            mode_choice = input(f"\n{Colors.BOLD}Select mode (0-3): {Colors.ENDC}")
            mode_choice = int(mode_choice)
            
            if mode_choice == 0:
                print_info("Exiting...")
                sys.exit(0)
            elif mode_choice in [1, 2, 3]:
                break
            else:
                print_warning("Please enter 0, 1, 2, or 3")
        except ValueError:
            print_warning("Invalid input. Please enter a number.")
    
    # Select checkpoint directory
    base_checkpoint_dir = Path("checkpoints")
    if not base_checkpoint_dir.exists():
        print_error(f"Checkpoint directory not found: {base_checkpoint_dir}")
        sys.exit(1)
    
    # First select training run
    run_dir = select_run_directory(base_checkpoint_dir)
    if run_dir is None:
        print_warning("No training run selected. Exiting.")
        sys.exit(0)
    
    if mode_choice == 1:
        # Single model mode
        checkpoint_path = select_checkpoint(run_dir)
        if checkpoint_path is None:
            print_info("Checkpoint selection cancelled.")
            sys.exit(0)
        
        # Load model and tokenizer
        print(f"\n{Colors.BOLD}Loading Model and Tokenizer...{Colors.ENDC}")
        try:
            model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            sys.exit(1)
        
        print_success("Model and tokenizer loaded successfully!")
        
        # Run single model inference loop
        run_single_model_inference(model, tokenizer, device)
        
    elif mode_choice == 2:
        # Multi-model mode
        checkpoint_paths = select_multiple_checkpoints(run_dir)
        if checkpoint_paths is None or len(checkpoint_paths) == 0:
            print_info("No checkpoints selected. Exiting.")
            sys.exit(0)
        
        # Memory warning for multiple models
        estimated_memory_per_model = 0.1  # ~100MB per model
        estimated_total_memory = len(checkpoint_paths) * estimated_memory_per_model
        
        if device.type == 'cuda':
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if estimated_total_memory > available_memory * 0.8:
                print_warning(f"Selected {len(checkpoint_paths)} models (~{estimated_total_memory:.1f}GB)")
                print_warning(f"Available GPU memory: {available_memory:.1f}GB")
                print_warning("This may exceed available memory!")
                confirm = input(f"\n{Colors.BOLD}Continue anyway? (y/n): {Colors.ENDC}")
                if confirm.lower() != 'y':
                    print_info("Model loading cancelled.")
                    sys.exit(0)
        else:
            if len(checkpoint_paths) > 5:
                print_warning(f"Loading {len(checkpoint_paths)} models on CPU may be slow.")
                confirm = input(f"\n{Colors.BOLD}Continue? (y/n): {Colors.ENDC}")
                if confirm.lower() != 'y':
                    print_info("Model loading cancelled.")
                    sys.exit(0)
        
        # Load multiple models
        print(f"\n{Colors.BOLD}Loading Multiple Models...{Colors.ENDC}")
        try:
            models_dict, total_memory = load_multiple_models(checkpoint_paths, device)
            if models_dict is None:
                print_error("Failed to load any models. Exiting.")
                sys.exit(1)
        except Exception as e:
            print_error(f"Failed to load models: {e}")
            if "out of memory" in str(e).lower():
                print_warning("Try selecting fewer models or using CPU instead.")
            sys.exit(1)
        
        print_success(f"Successfully loaded {len(models_dict)} models!")
        
        # Run multi-model inference loop
        run_multi_model_inference(models_dict, device)
    
    elif mode_choice == 3:
        # Benchmark mode
        checkpoint_paths = select_multiple_checkpoints(run_dir)
        if checkpoint_paths is None or len(checkpoint_paths) == 0:
            print_info("No checkpoints selected. Exiting.")
            sys.exit(0)
        
        # Memory warning for multiple models
        estimated_memory_per_model = 0.1  # ~100MB per model
        estimated_total_memory = len(checkpoint_paths) * estimated_memory_per_model
        
        if device.type == 'cuda':
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if estimated_total_memory > available_memory * 0.8:
                print_warning(f"Selected {len(checkpoint_paths)} models (~{estimated_total_memory:.1f}GB)")
                print_warning(f"Available GPU memory: {available_memory:.1f}GB")
                print_warning("This may exceed available memory!")
                confirm = input(f"\n{Colors.BOLD}Continue anyway? (y/n): {Colors.ENDC}")
                if confirm.lower() != 'y':
                    print_info("Model loading cancelled.")
                    sys.exit(0)
        else:
            if len(checkpoint_paths) > 3:
                print_warning(f"Loading {len(checkpoint_paths)} models on CPU for benchmarking may be slow.")
                confirm = input(f"\n{Colors.BOLD}Continue? (y/n): {Colors.ENDC}")
                if confirm.lower() != 'y':
                    print_info("Model loading cancelled.")
                    sys.exit(0)
        
        # Load multiple models for benchmarking
        print(f"\n{Colors.BOLD}Loading Models for Benchmarking...{Colors.ENDC}")
        try:
            models_dict, total_memory = load_multiple_models(checkpoint_paths, device)
            if models_dict is None:
                print_error("Failed to load any models. Exiting.")
                sys.exit(1)
        except Exception as e:
            print_error(f"Failed to load models: {e}")
            if "out of memory" in str(e).lower():
                print_warning("Try selecting fewer models or using CPU instead.")
            sys.exit(1)
        
        print_success(f"Successfully loaded {len(models_dict)} models for benchmarking!")
        
        # Run benchmark evaluation
        run_benchmark_mode(models_dict, device)


def run_single_model_inference(model: TransformerLM, tokenizer: WikipediaTokenizer, device: torch.device):
    """Run single model inference loop."""
    
    # Main generation loop
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.CYAN}Starting single model generation session...{Colors.ENDC}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    
    try:
        while True:
            # Select prompt mode
            mode, prompt = select_prompt_mode()
            
            if mode == 'exit':
                print_info("Exiting...")
                break
            
            if not prompt:
                print_warning("Empty prompt. Please try again.")
                continue
            
            # Get generation parameters (once for all prompts)
            print(f"\n{Colors.BOLD}Configure Generation:{Colors.ENDC}")
            print("  1. Use default parameters")
            print("  2. Customize parameters")
            
            param_choice = input(f"\n{Colors.BOLD}Select (1-2): {Colors.ENDC}")
            
            if param_choice == '2':
                generation_params = get_generation_params()
            else:
                generation_params = {
                    'max_length': 200,
                    'temperature': 0.8,
                    'top_k': 50,
                    'top_p': 0.95
                }
                print_info("Using default parameters")
            
            # Generate text
            print(f"\n{Colors.BOLD}Generating...{Colors.ENDC}")
            print("-" * 60)
            
            try:
                generated_text = generate_text(
                    model, tokenizer, prompt, device,
                    **generation_params
                )
                
                # Display generated text
                print(f"\n{Colors.GREEN}{Colors.BOLD}Generated Text:{Colors.ENDC}")
                print("=" * 60)
                print(generated_text)
                print("=" * 60)
                
            except Exception as e:
                print_error(f"Generation failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Ask to continue
            print(f"\n{Colors.BOLD}Continue?{Colors.ENDC}")
            cont = input("Press Enter to generate more, or 'q' to quit: ")
            if cont.lower() == 'q':
                break
    
    except KeyboardInterrupt:
        print_warning("\nInterrupted by user")
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_info("GPU cache cleared")
        
        print_success("Session ended. Goodbye!")










def run_benchmark_mode(models_dict: Dict[str, Tuple[TransformerLM, WikipediaTokenizer]], device: torch.device):
    """Run comprehensive benchmarks on selected models."""
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.CYAN}Starting benchmark evaluation...{Colors.ENDC}")
    print(f"{Colors.CYAN}{len(models_dict)} models loaded for benchmarking{Colors.ENDC}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    
    # Configure benchmark
    print(f"\n{Colors.BOLD}Configure Benchmark:{Colors.ENDC}")
    print("  1. Quick benchmark (fewer samples, faster)")
    print("  2. Standard benchmark (recommended)")
    print("  3. Comprehensive benchmark (more samples, slower)")
    
    while True:
        try:
            bench_choice = input(f"\n{Colors.BOLD}Select configuration (1-3): {Colors.ENDC}")
            bench_choice = int(bench_choice)
            
            if bench_choice == 1:
                config = {
                    'wikitext_samples': 50,
                    'lambada_samples': 20,
                    'hellaswag_samples': 10,
                    'max_length': 512,
                    'max_new_tokens': 20
                }
                print_info("Using quick benchmark configuration")
                break
            elif bench_choice == 2:
                config = {
                    'wikitext_samples': 100,
                    'lambada_samples': 50,
                    'hellaswag_samples': 20,
                    'max_length': 512,
                    'max_new_tokens': 30
                }
                print_info("Using standard benchmark configuration")
                break
            elif bench_choice == 3:
                config = {
                    'wikitext_samples': 200,
                    'lambada_samples': 100,
                    'hellaswag_samples': 40,
                    'max_length': 512,
                    'max_new_tokens': 40
                }
                print_info("Using comprehensive benchmark configuration")
                break
            else:
                print_warning("Please enter 1, 2, or 3")
        except ValueError:
            print_warning("Invalid input. Please enter a number.")
    
    # Run benchmarks
    print(f"\n{Colors.BOLD}Running Benchmarks...{Colors.ENDC}")
    print_warning("This may take several minutes depending on the configuration.")
    
    try:
        benchmark = ModelBenchmark(device)
        results = benchmark.run_full_benchmark(models_dict, config)
        
        # Display results
        print("\n" + format_benchmark_results(results))
        
        # Save results
        save_benchmark_results(results)
        
        print_success("\nBenchmark completed successfully!")
        
        # Ask if user wants to see detailed results
        print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
        print("  1. View detailed results per model")
        print("  2. Return to main menu")
        
        detail_choice = input(f"\n{Colors.BOLD}Select option (1-2): {Colors.ENDC}")
        
        if detail_choice == '1':
            for model_name, model_results in results.items():
                print(f"\n{Colors.BLUE}{'='*60}{Colors.ENDC}")
                print(f"{Colors.BLUE}Detailed Results for: {model_name}{Colors.ENDC}")
                print(f"{Colors.BLUE}{'='*60}{Colors.ENDC}")
                for metric, value in model_results.items():
                    if metric != 'error':
                        if isinstance(value, float):
                            if 'acc' in metric:
                                print(f"  {metric}: {value:.2f}%")
                            else:
                                print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
        
    except Exception as e:
        print_error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_info("GPU cache cleared")


def run_multi_model_inference(models_dict: Dict[str, Tuple[TransformerLM, WikipediaTokenizer]], device: torch.device):
    """Run multi-model comparison inference loop."""
    # Main generation loop
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.CYAN}Starting multi-model comparison session...{Colors.ENDC}")
    print(f"{Colors.CYAN}{len(models_dict)} models loaded for comparison{Colors.ENDC}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    
    try:
        while True:
            # Select prompt mode
            mode, prompt = select_prompt_mode()
            
            if mode == 'exit':
                print_info("Exiting...")
                break
            
            if not prompt:
                print_warning("Empty prompt. Please try again.")
                continue
            
            # Get generation parameters (once for all prompts and models)
            print(f"\n{Colors.BOLD}Configure Generation:{Colors.ENDC}")
            print("  1. Use default parameters")
            print("  2. Customize parameters")
            
            param_choice = input(f"\n{Colors.BOLD}Select (1-2): {Colors.ENDC}")
            
            if param_choice == '2':
                generation_params = get_generation_params()
            else:
                generation_params = {
                    'max_length': 200,
                    'temperature': 0.8,
                    'top_k': 50,
                    'top_p': 0.95
                }
                print_info("Using default parameters")
            
            # Generate from all models
            results = generate_text_multi_model(
                models_dict, prompt, device,
                **generation_params
            )
            
            # Display comparison results
            display_multi_model_results(results, prompt)
            
            # Ask to continue
            print(f"\n{Colors.BOLD}Continue?{Colors.ENDC}")
            cont = input("Press Enter to generate more, or 'q' to quit: ")
            if cont.lower() == 'q':
                break
    
    except KeyboardInterrupt:
        print_warning("\nInterrupted by user")
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_info("GPU cache cleared")
        
        print_success("Multi-model session ended. Goodbye!")


if __name__ == "__main__":
    main()