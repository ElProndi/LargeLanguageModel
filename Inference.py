#!/usr/bin/env python3
"""Interactive inference script for transformer language model."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Import project modules
from model import TransformerLM
from tokenizer import WikipediaTokenizer


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


def load_model_and_tokenizer(checkpoint_path: Path, device: torch.device) -> Tuple[TransformerLM, WikipediaTokenizer]:
    """Load model from checkpoint and tokenizer.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print_info("Loading configuration...")
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    # Create model
    print_info("Creating model architecture...")
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        max_position_embeddings=config['model']['max_position_embeddings'],
        dropout=0.0,  # Disable dropout for inference
        attention_dropout=0.0,  # Disable attention dropout for inference
        layer_norm_eps=config['model']['layer_norm_eps'],
        initializer_range=config['model']['initializer_range'],
        use_cache=config['model']['use_cache'],
        pad_token_id=config['tokenizer'].get('pad_token_id', 3)
    )
    
    # Load checkpoint
    print_info(f"Loading checkpoint: {checkpoint_path.name}...")
    # Note: weights_only=False is safe here since we're loading our own trained checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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
    
    # Try to load full tokenizer first, then fallback to test tokenizer
    tokenizer_paths = [
        Path("tokenizers/full_tokenizer"),
        Path("tokenizers/test_tokenizer")
    ]
    
    tokenizer_loaded = False
    for tokenizer_path in tokenizer_paths:
        if tokenizer_path.exists():
            try:
                tokenizer.load(str(tokenizer_path))
                print_success(f"Tokenizer loaded from {tokenizer_path}")
                tokenizer_loaded = True
                break
            except Exception as e:
                print_warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
    
    if not tokenizer_loaded:
        print_error("No tokenizer found! Please train a tokenizer first using tokenizer.py")
        sys.exit(1)
    
    return model, tokenizer


def get_predefined_prompts() -> List[str]:
    """Get list of predefined prompts for generation."""
    return [
        "Once upon a time",
        "The future of artificial intelligence",
        "In the year 2050",
        "Scientists have discovered",
        "Breaking news:",
        "The most important thing in life is",
        "Today I learned that",
        "The secret to happiness is",
        "In a galaxy far, far away",
        "The recipe for success includes"
    ]


def select_prompt_mode() -> Tuple[str, Optional[str]]:
    """Select prompt input mode.
    
    Returns:
        Tuple of (mode, prompt) where mode is 'custom' or 'predefined'
    """
    print(f"\n{Colors.BOLD}Select Input Mode:{Colors.ENDC}")
    print("  1. Write custom prompt")
    print("  2. Use predefined prompts")
    print("  0. Exit")
    
    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Select mode (0-2): {Colors.ENDC}")
            choice = int(choice)
            
            if choice == 0:
                return 'exit', None
            elif choice == 1:
                prompt = input(f"\n{Colors.BOLD}Enter your prompt: {Colors.ENDC}")
                return 'custom', prompt
            elif choice == 2:
                prompts = get_predefined_prompts()
                print(f"\n{Colors.BOLD}Predefined Prompts:{Colors.ENDC}")
                for idx, prompt in enumerate(prompts, 1):
                    print(f"  {idx:2d}. {prompt}")
                print(f"\n  {Colors.CYAN}Press Enter to use ALL prompts{Colors.ENDC}")
                
                while True:
                    try:
                        prompt_choice = input(f"\n{Colors.BOLD}Select prompt (1-{len(prompts)} or Enter for all): {Colors.ENDC}")
                        
                        # If empty input, return all prompts
                        if not prompt_choice.strip():
                            print_success("Using all predefined prompts")
                            return 'predefined_all', prompts  # Return all prompts
                        
                        prompt_idx = int(prompt_choice) - 1
                        if 0 <= prompt_idx < len(prompts):
                            return 'predefined', prompts[prompt_idx]
                        else:
                            print_warning(f"Please enter a number between 1 and {len(prompts)}")
                    except ValueError:
                        print_warning("Invalid input. Please enter a number or press Enter for all.")
            else:
                print_warning("Please enter 0, 1, or 2")
        except (ValueError, KeyboardInterrupt):
            print_warning("Invalid input.")


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
    # Encode prompt
    print_info(f"Encoding prompt...")
    input_ids = tokenizer.encode(prompt)
    
    # Add BOS token at the beginning if not present
    bos_id = tokenizer.tokenizer.token_to_id("<BOS>")
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
            eos_token_id=tokenizer.tokenizer.token_to_id("<EOS>"),
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
    
    # Select checkpoint
    base_checkpoint_dir = Path("checkpoints")
    if not base_checkpoint_dir.exists():
        print_error(f"Checkpoint directory not found: {base_checkpoint_dir}")
        sys.exit(1)
    
    # First select training run
    run_dir = select_run_directory(base_checkpoint_dir)
    if run_dir is None:
        print_warning("No training run selected. Exiting.")
        sys.exit(0)
    
    # Then select checkpoint within the run
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
    
    # Main generation loop
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.CYAN}Starting interactive generation session...{Colors.ENDC}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    
    try:
        while True:
            # Select prompt mode
            mode, prompt = select_prompt_mode()
            
            if mode == 'exit':
                print_info("Exiting...")
                break
            
            # Handle single prompt or list of prompts
            if mode == 'predefined_all':
                prompts_to_generate = prompt  # prompt is actually a list in this case
            else:
                if not prompt:
                    print_warning("Empty prompt. Please try again.")
                    continue
                prompts_to_generate = [prompt]  # Convert single prompt to list
            
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
            
            # Generate text for each prompt
            for idx, current_prompt in enumerate(prompts_to_generate, 1):
                if len(prompts_to_generate) > 1:
                    print(f"\n{Colors.BLUE}{'='*60}{Colors.ENDC}")
                    print(f"{Colors.BLUE}{Colors.BOLD}Prompt {idx}/{len(prompts_to_generate)}: {current_prompt}{Colors.ENDC}")
                    print(f"{Colors.BLUE}{'='*60}{Colors.ENDC}")
                
                print(f"\n{Colors.BOLD}Generating...{Colors.ENDC}")
                print("-" * 60)
                
                try:
                    generated_text = generate_text(
                        model, tokenizer, current_prompt, device,
                        **generation_params
                    )
                    
                    # Display generated text
                    print(f"\n{Colors.GREEN}{Colors.BOLD}Generated Text:{Colors.ENDC}")
                    print("=" * 60)
                    print(generated_text)
                    print("=" * 60)
                    
                except Exception as e:
                    print_error(f"Generation failed for prompt '{current_prompt}': {e}")
                    if len(prompts_to_generate) > 1:
                        print_info("Continuing with next prompt...")
                        continue
                    else:
                        import traceback
                        traceback.print_exc()
            
            # Show summary if multiple prompts were generated
            if len(prompts_to_generate) > 1:
                print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
                print(f"{Colors.CYAN}{Colors.BOLD}Batch Generation Complete!{Colors.ENDC}")
                print(f"{Colors.CYAN}Generated text for {len(prompts_to_generate)} prompts{Colors.ENDC}")
                print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
            
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


if __name__ == "__main__":
    main()