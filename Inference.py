#!/usr/bin/env python3
"""Interactive inference script for transformer language model."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

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


def print_message(text: str, style: str = "info", symbol: Optional[str] = None):
    """Unified print function for all message types.
    
    Args:
        text: Message text to print
        style: Style of message ('header', 'info', 'success', 'warning', 'error', 'cyan', 'blue')
        symbol: Optional symbol to prefix the message (auto-selected if None)
    """
    style_config = {
        'header': {'color': Colors.HEADER, 'symbol': '', 'bold': True, 'is_header': True},
        'info': {'color': Colors.CYAN, 'symbol': 'ℹ', 'bold': False, 'is_header': False},
        'success': {'color': Colors.GREEN, 'symbol': '✓', 'bold': False, 'is_header': False},
        'warning': {'color': Colors.WARNING, 'symbol': '⚠', 'bold': False, 'is_header': False},
        'error': {'color': Colors.RED, 'symbol': '✗', 'bold': False, 'is_header': False},
        'cyan': {'color': Colors.CYAN, 'symbol': '', 'bold': False, 'is_header': False},
        'blue': {'color': Colors.BLUE, 'symbol': '', 'bold': False, 'is_header': False}
    }
    
    config = style_config.get(style, style_config['info'])
    color = config['color']
    default_symbol = config['symbol']
    use_bold = config['bold']
    is_header = config['is_header']
    
    # Use provided symbol or default for the style
    display_symbol = symbol if symbol is not None else default_symbol
    
    if is_header:
        # Special formatting for headers
        print(f"\n{color}{Colors.BOLD if use_bold else ''}{'=' * 60}{Colors.ENDC}")
        print(f"{color}{Colors.BOLD if use_bold else ''}{text.center(60)}{Colors.ENDC}")
        print(f"{color}{Colors.BOLD if use_bold else ''}{'=' * 60}{Colors.ENDC}\n")
    else:
        # Regular message formatting
        prefix = f"{display_symbol} " if display_symbol else ""
        bold_start = Colors.BOLD if use_bold else ""
        print(f"{color}{bold_start}{prefix}{text}{Colors.ENDC}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


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


def select_items_interactive(
    items: List[Tuple[str, any]],
    title: str,
    allow_multiple: bool = False,
    allow_all: bool = False,
    allow_skip: bool = False,
    special_highlights: Optional[Dict[str, str]] = None
) -> Optional[Union[any, List[any]]]:
    """Generic interactive item selection function.
    
    Args:
        items: List of (display_name, value) tuples
        title: Title to display
        allow_multiple: Allow selecting multiple items
        allow_all: Show 'all' option for selecting all items
        allow_skip: Show 'skip' option (only valid in multiple mode)
        special_highlights: Dict mapping keywords to highlight colors (e.g., {'best': 'GREEN'})
    
    Returns:
        Selected value(s) or None if cancelled
    """
    if not items:
        print_message(f"No items available for selection", "error")
        return None
    
    # Display header
    print(f"{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 70)
    
    # Display items with highlighting
    for idx, (display_name, _) in enumerate(items, 1):
        if special_highlights:
            for keyword, color_name in special_highlights.items():
                if keyword in display_name.lower():
                    color = getattr(Colors, color_name, Colors.ENDC)
                    if keyword == 'most recent' and idx == 1:
                        print(f"  {color}{idx:2d}. {display_name} (most recent){Colors.ENDC}")
                    else:
                        print(f"  {color}{idx:2d}. {display_name}{Colors.ENDC}")
                    break
            else:
                print(f"  {idx:2d}. {display_name}")
        else:
            print(f"  {idx:2d}. {display_name}")
    
    print("-" * 70)
    
    # Display options
    if allow_multiple:
        print(f"{Colors.CYAN}Enter numbers separated by commas (e.g., 1,3,5){Colors.ENDC}")
        if allow_all:
            print(f"{Colors.CYAN}Or enter 'all' to select all items{Colors.ENDC}")
        if allow_skip:
            print(f"{Colors.CYAN}Or enter 'skip' to skip this selection{Colors.ENDC}")
    print(f"{Colors.WARNING}Enter 0 to cancel{Colors.ENDC}")
    
    # Get user input
    while True:
        try:
            if allow_multiple:
                prompt = f"\n{Colors.BOLD}Select items: {Colors.ENDC}"
            else:
                prompt = f"\n{Colors.BOLD}Select item (0-{len(items)}): {Colors.ENDC}"
            
            choice = input(prompt).strip()
            
            # Handle cancel
            if choice == '0':
                return None
            
            # Handle special options for multiple selection
            if allow_multiple:
                if allow_skip and choice.lower() == 'skip':
                    return 'skip'  # Special return value for skip
                
                if allow_all and choice.lower() == 'all':
                    selected_values = [value for _, value in items]
                    print_message(f"Selected all {len(selected_values)} items", "success")
                    return selected_values
                
                # Parse comma-separated numbers
                try:
                    indices = [int(x.strip()) for x in choice.split(',') if x.strip()]
                    selected_values = []
                    selected_names = []
                    
                    for idx in indices:
                        if 1 <= idx <= len(items):
                            name, value = items[idx - 1]
                            selected_values.append(value)
                            selected_names.append(name.split()[0] if len(name.split()) > 0 else name)
                        else:
                            print_message(f"Invalid index {idx}, skipping...", "warning")
                    
                    if selected_values:
                        print_message(f"Selected {len(selected_values)} items", "success")
                        for name in selected_names:
                            print(f"  • {name}")
                        return selected_values
                    else:
                        print_message("No valid items selected. Please try again.", "warning")
                        
                except ValueError:
                    if allow_skip:
                        print_message("Invalid input. Enter numbers, 'all', 'skip', or 0 to cancel.", "warning")
                    else:
                        print_message("Invalid input. Enter numbers, 'all', or 0 to cancel.", "warning")
            
            else:
                # Single selection mode
                try:
                    choice_idx = int(choice)
                    if 1 <= choice_idx <= len(items):
                        name, value = items[choice_idx - 1]
                        print_message(f"Selected: {name.split()[0] if len(name.split()) > 0 else name}", "success")
                        return value
                    else:
                        print_message(f"Please enter a number between 0 and {len(items)}", "warning")
                except ValueError:
                    print_message("Invalid input. Please enter a number.", "warning")
                    
        except KeyboardInterrupt:
            print_message("\nSelection cancelled", "warning")
            return None


def select_checkpoints_unified(base_checkpoint_dir: Path, allow_multiple: bool = False) -> Optional[Union[Path, List[Tuple[Path, Path]]]]:
    """Unified checkpoint selection for single or multiple models.
    
    Returns:
        For single: checkpoint path
        For multiple: list of (run_dir, checkpoint_path) tuples
    """
    run_dirs = list_run_directories(base_checkpoint_dir)
    
    # Handle flat structure (backward compatibility)
    if not run_dirs:
        if any(base_checkpoint_dir.glob("*.pt")):
            print_message("Found checkpoints in old flat structure.", "warning")
            checkpoints = list_checkpoints(base_checkpoint_dir)
            if not checkpoints:
                print_message("No checkpoints found.", "error")
                return None
            
            result = select_items_interactive(
                checkpoints,
                "Available Checkpoints:",
                allow_multiple=allow_multiple,
                allow_all=allow_multiple,
                special_highlights={'best': 'GREEN', 'latest': 'BLUE'}
            )
            
            if result is None:
                return None
            
            if allow_multiple:
                return [(base_checkpoint_dir, cp) for cp in result]
            else:
                return result
        else:
            print_message(f"No training runs found in {base_checkpoint_dir}", "error")
            return None
    
    # Single model selection
    if not allow_multiple:
        run_dir = select_items_interactive(
            run_dirs,
            "Available Training Runs:",
            special_highlights={'most recent': 'GREEN'}
        )
        if run_dir is None:
            return None
        
        checkpoints = list_checkpoints(run_dir)
        if not checkpoints:
            print_message(f"No checkpoints found in {run_dir}", "error")
            return None
        
        return select_items_interactive(
            checkpoints,
            "Available Checkpoints:",
            special_highlights={'best': 'GREEN', 'latest': 'BLUE'}
        )
    
    # Multi-model selection
    print(f"{Colors.BOLD}Step 1: Select Training Runs{Colors.ENDC}")
    selected_runs = select_items_interactive(
        run_dirs,
        "Select Training Runs:",
        allow_multiple=True,
        allow_all=True,
        special_highlights={'most recent': 'GREEN'}
    )
    
    if not selected_runs:
        return None
    
    all_selected = []
    for run_idx, run_path in enumerate(selected_runs, 1):
        print(f"\n{Colors.BOLD}Step 2.{run_idx}: Select from {run_path.name}{Colors.ENDC}")
        
        checkpoints = list_checkpoints(run_path)
        if not checkpoints:
            print_message(f"No checkpoints in {run_path.name}", "warning")
            continue
        
        selected = select_items_interactive(
            checkpoints,
            f"Checkpoints in {run_path.name}:",
            allow_multiple=True,
            allow_all=True,
            allow_skip=True,
            special_highlights={'best': 'GREEN', 'latest': 'BLUE'}
        )
        
        if selected == 'skip':
            continue
        elif selected:
            for cp in selected:
                all_selected.append((run_path, cp))
    
    if all_selected:
        print_message(f"\nTotal selected: {len(all_selected)} checkpoints", "success")
        return all_selected
    
    return None


def check_memory_constraints(num_models: int, device: torch.device, model_type: str = "inference") -> bool:
    """Check memory constraints and warn user if needed.
    
    Args:
        num_models: Number of models to load
        device: Device to load models on
        model_type: Type of operation ("inference", "benchmark", etc.)
    
    Returns:
        True if user wants to continue, False otherwise
    """
    estimated_memory_per_model = 0.1  # ~100MB per model
    estimated_total_memory = num_models * estimated_memory_per_model
    
    if device.type == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if estimated_total_memory > available_memory * 0.8:
            print_message(f"Selected {num_models} models (~{estimated_total_memory:.1f}GB)", "warning")
            print_message(f"Available GPU memory: {available_memory:.1f}GB", "warning")
            print_message("This may exceed available memory!", "warning")
            confirm = input(f"\n{Colors.BOLD}Continue anyway? (y/n): {Colors.ENDC}")
            return confirm.lower() == 'y'
    else:
        # Different thresholds for different operations
        cpu_threshold = 3 if model_type == "benchmark" else 5
        if num_models > cpu_threshold:
            print_message(f"Loading {num_models} models on CPU for {model_type} may be slow.", "warning")
            confirm = input(f"\n{Colors.BOLD}Continue? (y/n): {Colors.ENDC}")
            return confirm.lower() == 'y'
    
    return True  # No memory constraints


def create_model_metadata(model: TransformerLM, run_name: str, checkpoint_path: Path, memory_bytes: float) -> Dict:
    """Create standardized metadata for a model."""
    return {
        'run': run_name,
        'checkpoint': checkpoint_path.name,
        'params': sum(p.numel() for p in model.parameters()),
        'memory_mb': memory_bytes / 1e6
    }

def load_models_for_inference(base_checkpoint_dir: Path, device: torch.device, multi_model: bool = False, compile_model: bool = True) -> Optional[Tuple[Dict, Optional[Dict]]]:
    """Load model(s) for inference.
    
    Args:
        base_checkpoint_dir: Base directory for checkpoints
        device: Device to load models on
        multi_model: Whether to load multiple models
        compile_model: Whether to compile models with torch.compile() (default: True)
    
    Returns:
        Tuple of (models_dict, metadata) or None if cancelled
    """
    # Select checkpoints
    selection = select_checkpoints_unified(base_checkpoint_dir, allow_multiple=multi_model)
    if selection is None:
        return None
    
    if not multi_model:
        # Single model - selection is a Path
        checkpoint_path = selection
        print(f"\n{Colors.BOLD}Loading Model and Tokenizer...{Colors.ENDC}")
        try:
            model, tokenizer = load_model_and_tokenizer(checkpoint_path, device, compile_model)
            models_dict = {checkpoint_path.name: (model, tokenizer)}
            memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            run_name = checkpoint_path.parent.name if checkpoint_path.parent != base_checkpoint_dir else "base"
            metadata = {
                checkpoint_path.name: create_model_metadata(model, run_name, checkpoint_path, memory_bytes)
            }
            print_message("Model and tokenizer loaded successfully!", "success")
            return models_dict, metadata
        except Exception as e:
            print_message(f"Failed to load model: {e}", "error")
            return None
    else:
        # Multi-model - selection is a list of (run_dir, checkpoint_path) tuples
        checkpoint_info = selection
        if not check_memory_constraints(len(checkpoint_info), device, "inference"):
            return None
        
        print(f"\n{Colors.BOLD}Loading Multiple Models...{Colors.ENDC}")
        try:
            models_dict, metadata, _ = load_multiple_models(checkpoint_info, device, compile_model)
            if models_dict:
                print_message(f"Successfully loaded {len(models_dict)} models!", "success")
                return models_dict, metadata
            return None
        except Exception as e:
            print_message(f"Failed to load models: {e}", "error")
            return None


def _load_tokenizer() -> WikipediaTokenizer:
    """Load or download the CodeLlama tokenizer.
    
    Returns:
        Loaded WikipediaTokenizer instance
    """
    print_message("Loading tokenizer...", "info")
    tokenizer = WikipediaTokenizer()
    tokenizer_path = Path("tokenizers/codellama_tokenizer")
    
    if tokenizer_path.exists():
        try:
            tokenizer.load(str(tokenizer_path))
            print_message(f"CodeLlama tokenizer loaded from {tokenizer_path}", "success")
        except Exception as e:
            print_message(f"Failed to load tokenizer from {tokenizer_path}: {e}", "error")
            sys.exit(1)
    else:
        print_message("CodeLlama tokenizer not found locally. Downloading from HuggingFace...", "info")
        tokenizer.train()  # Downloads the pre-trained tokenizer
        tokenizer.save(str(tokenizer_path))
        print_message(f"CodeLlama tokenizer downloaded and saved to {tokenizer_path}", "success")
    
    return tokenizer


def load_model_and_tokenizer(checkpoint_path: Path, device: torch.device, compile_model: bool = True) -> Tuple[TransformerLM, WikipediaTokenizer]:
    """Load model from checkpoint and tokenizer.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        compile_model: Whether to compile the model with torch.compile() (default: True)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load checkpoint first to get the config
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
    use_scaled_residuals = False
    
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
    
    # Check for residual scaling in checkpoint
    use_scaled_residuals = any('residual_scale' in key for key in state_dict.keys())
    
    # Read Flash Attention setting from checkpoint config (required)
    if 'use_flash_attention' not in architecture_features:
        raise ValueError(
            f"Missing 'use_flash_attention' in checkpoint's architecture_features. "
            f"Available features: {list(architecture_features.keys())}"
        )
    
    use_flash_attention_inference = architecture_features['use_flash_attention']
    
    # Disable KV caching when Flash Attention is enabled (they're mutually exclusive)
    use_kv_cache = not use_flash_attention_inference
    
    # Log the settings being used
    print(f"  Flash Attention: {'✓ Enabled' if use_flash_attention_inference else '✗ Disabled'}")
    print(f"  KV Caching: {'✓ Enabled' if use_kv_cache else '✗ Disabled (Flash Attention active)'}")
    
    # Create model with all architecture parameters from config
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        # CRITICAL: Pass num_kv_heads to match trained model architecture
        num_kv_heads=model_config.get('num_kv_heads', model_config['num_heads']),
        # Pass detected or configured intermediate_size
        intermediate_size=intermediate_size,  # Use detected size from checkpoint
        max_position_embeddings=model_config['max_position_embeddings'],
        # RoPE configuration
        rope_theta=rope_config.get('theta', 10000.0),
        rope_scaling=rope_config.get('scaling_factor', 1.0),
        # Use detected residual scaling setting
        use_scaled_residuals=use_scaled_residuals,
        # Architecture features from config - CRITICAL for correct model reconstruction
        use_gqa=architecture_features.get('use_gqa', True),
        use_flash_attention=use_flash_attention_inference,  # Use inference-specific setting
        tie_embeddings=architecture_features.get('tie_embeddings', True),
        use_gradient_checkpointing=architecture_features.get('use_gradient_checkpointing', False),
        # Disable dropout for inference
        dropout=0.0,
        attention_dropout=0.0,
        layer_norm_eps=model_config['layer_norm_eps'],
        initializer_range=model_config['initializer_range'],
        use_cache=use_kv_cache,  # Disabled when Flash Attention is active
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
    
    # Conditionally compile model based on parameter
    if compile_model:
        print("  Compiling model with torch.compile()...")
        model = torch.compile(model)
        print("  Model compiled successfully")
    else:
        print("  Compilation: ✗ Disabled (benchmark mode - avoids Flash Attention issues)")
    
    # Get model stats
    total_params = sum(p.numel() for p in model.parameters())
    print_message(f"Model loaded: {total_params:,} parameters", "success")
    
    
    # Load tokenizer (shared across all models)
    tokenizer = _load_tokenizer()
    
    return model, tokenizer


def load_multiple_models(
    checkpoint_info: Union[List[Path], List[Tuple[Path, Path]]], 
    device: torch.device,
    compile_model: bool = True
) -> Tuple[Dict[str, Tuple[TransformerLM, WikipediaTokenizer]], Dict[str, Dict], float]:
    """Load multiple models from checkpoints.
    
    Args:
        checkpoint_info: Either a list of checkpoint paths (old format)
                        or a list of (run_dir, checkpoint_path) tuples (new format)
        device: Device to load models on
        compile_model: Whether to compile models with torch.compile() (default: True)
    
    Returns:
        Tuple of (models_dict, metadata_dict, total_memory_gb)
    """
    models = {}
    metadata = {}
    total_memory = 0
    
    # Check available memory first
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print_message(f"Available GPU memory: {free_memory / 1e9:.2f} GB", "info")
    
    # Normalize checkpoint format
    if checkpoint_info and isinstance(checkpoint_info[0], tuple):
        checkpoint_list = [(r.name if r != c.parent else "base", c) for r, c in checkpoint_info]
    else:
        checkpoint_list = [("base", c) for c in checkpoint_info]
    
    print(f"\n{Colors.BOLD}Loading {len(checkpoint_list)} Models...{Colors.ENDC}")
    print("-" * 60)
    
    # Load shared tokenizer once
    tokenizer = _load_tokenizer()
    
    for idx, (run_name, checkpoint_path) in enumerate(checkpoint_list, 1):
        model_name = f"{run_name}/{checkpoint_path.name}" if run_name != "base" else checkpoint_path.name
        print(f"\n{Colors.CYAN}Model {idx}/{len(checkpoint_list)}: {model_name}{Colors.ENDC}")
        
        try:
            # Load only the model (tokenizer is shared)
            model, _ = load_model_and_tokenizer(checkpoint_path, device, compile_model)
            
            # Calculate memory
            model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            total_memory += model_memory
            
            # Store model
            models[model_name] = (model, tokenizer)
            metadata[model_name] = create_model_metadata(model, run_name, checkpoint_path, model_memory)
            
            print_message(f"✓ Loaded {model_name} ({model_memory / 1e6:.1f} MB)", "success")
            
            if device.type == 'cuda':
                print_message(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB", "info")
                
        except Exception as e:
            print_message(f"Failed to load {model_name}: {e}", "error")
            print_message(f"Continuing with other models...", "warning")
    
    print("\n" + "=" * 60)
    print_message(f"Loaded {len(models)}/{len(checkpoint_list)} models", "success")
    print_message(f"Total memory: {total_memory / 1e9:.2f} GB", "info")
    
    return models, metadata, total_memory / 1e9 if models else (None, None, 0)








def encode_prompt(tokenizer: WikipediaTokenizer, prompt: str, device: torch.device) -> Tuple[torch.Tensor, List[int], Dict]:
    """Encode prompt text into tokens with proper handling.
    
    Args:
        tokenizer: Tokenizer to use
        prompt: Text prompt
        device: Device to place tensor on
    
    Returns:
        Tuple of (input_tensor, input_ids, special_tokens)
    """
    # Encode prompt without auto-added special tokens to avoid trailing EOS
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Get special token IDs
    special_tokens = tokenizer.get_special_token_ids()
    bos_id = special_tokens['bos_token_id']
    eos_id = special_tokens['eos_token_id']
    
    # Add BOS token explicitly (training data always had BOS at start)
    input_ids = [bos_id] + input_ids
    
    # Defensive: ensure no EOS appended by mistake
    if len(input_ids) > 0 and input_ids[-1] == eos_id:
        input_ids = input_ids[:-1]
    
    # Convert to tensor
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    return input_tensor, input_ids, special_tokens


def calculate_generation_stats(generated_ids: List[int], input_ids: List[int], generation_time: float) -> Dict[str, any]:
    """Calculate generation statistics.
    
    Args:
        generated_ids: Generated token IDs
        input_ids: Input token IDs
        generation_time: Time taken for generation
    
    Returns:
        Dictionary with generation statistics
    """
    tokens_generated = len(generated_ids) - len(input_ids)
    tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
    
    return {
        'tokens_generated': tokens_generated,
        'generation_time': generation_time,
        'tokens_per_sec': tokens_per_sec,
        'input_tokens': len(input_ids)
    }


def stream_tokens(tokenizer: WikipediaTokenizer, token_generator) -> Tuple[str, Dict]:
    """Stream tokens to console as they're generated.
    
    Args:
        tokenizer: Tokenizer to decode tokens
        token_generator: Generator that yields token IDs
    
    Returns:
        Tuple of (complete text, statistics)
    """
    import sys
    
    generated_tokens = []
    token_count = 0
    start_time = time.time()
    
    # Track previously decoded text length for incremental decoding
    previous_text_length = 0
    
    for item in token_generator:
        # Check if this is the final statistics dict
        if isinstance(item, dict) and item.get('complete'):
            # Final statistics
            generation_time = time.time() - start_time
            stats = {
                'tokens_generated': token_count,
                'generation_time': generation_time,
                'tokens_per_sec': token_count / generation_time if generation_time > 0 else 0
            }
            # Decode complete sequence if available
            if 'full_sequence' in item:
                full_text = tokenizer.decode(item['full_sequence'][0].tolist())
            else:
                # Fallback to accumulated tokens
                full_text = tokenizer.decode(generated_tokens)
            return full_text, stats
        else:
            # Regular token - add to our accumulated list
            generated_tokens.append(item)
            token_count += 1
            
            # Incremental decoding: decode the full sequence to get proper text
            # This ensures spaces and special characters are properly reconstructed
            full_decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            # Extract only the new portion of text that was added
            new_text = full_decoded_text[previous_text_length:]
            
            # Print the new text portion and flush immediately for real-time display
            if new_text:  # Only write if there's actual new text
                sys.stdout.write(new_text)
                sys.stdout.flush()
                
            # Update the previous text length for next iteration
            previous_text_length = len(full_decoded_text)
    
    # If generator ended without sending stats (shouldn't happen)
    generation_time = time.time() - start_time
    stats = {
        'tokens_generated': token_count,
        'generation_time': generation_time,
        'tokens_per_sec': token_count / generation_time if generation_time > 0 else 0
    }
    full_text = tokenizer.decode(generated_tokens) if generated_tokens else ""
    return full_text, stats


def generate_text_unified(
    models_dict: Dict[str, Tuple[TransformerLM, WikipediaTokenizer]],
    prompt: str,
    device: torch.device,
    stream: bool = True,
    **generation_params
) -> Union[str, Dict[str, Dict[str, any]]]:
    """Unified text generation for single or multiple models with streaming support.
    
    Args:
        models_dict: Dictionary of model name to (model, tokenizer) tuples
        prompt: Text prompt to generate from
        device: Device to run on
        stream: Whether to stream tokens as they're generated
        **generation_params: Generation parameters (max_length, temperature, etc.)
    
    Returns:
        str for single model, dict of results for multiple models
    """
    single_model = len(models_dict) == 1
    results = {} if not single_model else None
    
    if not single_model:
        print(f"\n{Colors.BOLD}Generating from {len(models_dict)} models...{Colors.ENDC}")
        print(f"Prompt: {Colors.CYAN}{prompt}{Colors.ENDC}")
        print("-" * 60)
    
    for idx, (model_name, (model, tokenizer)) in enumerate(models_dict.items(), 1):
        if not single_model:
            print(f"\n{Colors.BLUE}[{idx}/{len(models_dict)}] Model: {model_name}{Colors.ENDC}")
        
        try:
            # Encode prompt
            if single_model:
                print_message("Encoding prompt...", "info")
            input_tensor, input_ids, special_tokens = encode_prompt(tokenizer, prompt, device)
            print_message(f"Input tokens: {len(input_ids)}", "info")
            
            if stream and hasattr(model, 'generate_stream'):
                # Streaming generation
                print_message("Generating (streaming)...", "info")
                print(f"\n{Colors.GREEN}Generated Text:{Colors.ENDC}\n{prompt}", end="")
                
                # Create generator
                token_generator = model.generate_stream(
                    input_tensor,
                    eos_token_id=special_tokens['eos_token_id'],
                    **generation_params
                )
                
                # Stream tokens and get final text
                generated_text, stats = stream_tokens(tokenizer, token_generator)
                
                print()  # New line after streaming
                print_message("\nGeneration complete!", "success")
                print_message(f"Tokens generated: {stats['tokens_generated']}", "info")
                print_message(f"Time taken: {stats['generation_time']:.2f}s", "info")
                print_message(f"Speed: {stats['tokens_per_sec']:.1f} tokens/sec", "info")
                
            else:
                # Non-streaming generation (fallback)
                print_message("Generating...", "info")
                start_time = time.time()
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_tensor,
                        eos_token_id=special_tokens['eos_token_id'],
                        **generation_params
                    )
                generation_time = time.time() - start_time
                
                # Decode and calculate stats
                generated_ids = generated_ids[0].tolist()
                generated_text = tokenizer.decode(generated_ids)
                stats = calculate_generation_stats(generated_ids, input_ids, generation_time)
                
                print_message("Generation complete!", "success")
                print_message(f"Tokens generated: {stats['tokens_generated']}", "info")
                print_message(f"Time taken: {stats['generation_time']:.2f}s", "info")
                print_message(f"Speed: {stats['tokens_per_sec']:.1f} tokens/sec", "info")
            
            if single_model:
                return generated_text
            else:
                results[model_name] = {'text': generated_text, **stats}
                
        except Exception as e:
            if single_model:
                raise e
            print_message(f"Generation failed for {model_name}: {e}", "error")
            results[model_name] = {
                'text': f"[ERROR: {str(e)}]",
                'tokens_generated': 0,
                'generation_time': 0,
                'tokens_per_sec': 0,
                'input_tokens': 0,
                'error': True
            }
    
    return results


def get_general_knowledge_prompts() -> List[Tuple[str, str]]:
    """Get a curated list of general knowledge text completion prompts.
    
    Returns:
        List of (category, prompt) tuples
    """
    prompts = [
        ("Science - Physics", "The theory of relativity was developed by Albert Einstein and states that"),
        ("Science - Biology", "Photosynthesis is the process by which plants"),
        ("History", "The Renaissance period in Europe was characterized by"),
        ("Geography", "The Amazon rainforest is important for the global climate because"),
        ("Mathematics", "The Pythagorean theorem states that in a right triangle"),
        ("Technology", "Artificial intelligence differs from traditional programming in that"),
        ("Literature", "Shakespeare's most famous works include plays such as"),
        ("Chemistry", "The periodic table organizes elements based on"),
        ("Economics", "Supply and demand are fundamental concepts that explain how"),
        ("Philosophy", "The Socratic method is a form of inquiry that involves")
    ]
    return prompts


def save_general_knowledge_results(
    all_results: Dict[str, List[Dict[str, any]]],
    metadata: Optional[Dict[str, Dict]] = None,
    output_dir: str = "general_knowledge_results"
) -> None:
    """Save general knowledge evaluation results to file.
    
    Args:
        all_results: Dictionary mapping model names to lists of result dicts
        metadata: Optional model metadata
        output_dir: Directory to save results in
    """
    from pathlib import Path
    from datetime import datetime
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full JSON results (untruncated)
    json_filename = output_path / f"general_knowledge_{timestamp}.json"
    full_results = {
        'timestamp': timestamp,
        'models': list(all_results.keys()),
        'metadata': metadata,
        'results': all_results,
        'summary': {}
    }
    
    # Calculate summary statistics
    for model_name, results_list in all_results.items():
        successful = [r for r in results_list if not r.get('error', False)]
        if successful:
            full_results['summary'][model_name] = {
                'success_rate': len(successful) / len(results_list),
                'avg_tokens': sum(r['tokens_generated'] for r in successful) / len(successful),
                'avg_time': sum(r['generation_time'] for r in successful) / len(successful),
                'total_tokens': sum(r['tokens_generated'] for r in successful)
            }
    
    with open(json_filename, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print_message(f"Full results saved to: {json_filename}", "info")
    
    # Save human-readable text version (full text, no truncation)
    text_filename = output_path / f"general_knowledge_{timestamp}.txt"
    with open(text_filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENERAL KNOWLEDGE EVALUATION - FULL RESULTS\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write model information
        f.write("MODELS EVALUATED:\n")
        f.write("-" * 80 + "\n")
        for model_name in all_results.keys():
            f.write(f"• {model_name}")
            if metadata and model_name in metadata:
                meta = metadata[model_name]
                f.write(f" ({meta['params']:,} parameters, {meta['memory_mb']:.1f} MB)")
            f.write("\n")
        f.write("\n")
        
        # Write detailed results for each category (NO TRUNCATION)
        f.write("DETAILED RESULTS BY CATEGORY:\n")
        f.write("=" * 80 + "\n\n")
        
        # Group results by category
        categories_seen = []
        for results_list in all_results.values():
            for result in results_list:
                if result['category'] not in categories_seen:
                    categories_seen.append(result['category'])
        
        for category in categories_seen:
            f.write(f"CATEGORY: {category}\n")
            f.write("-" * 80 + "\n")
            
            # Get the prompt for this category
            prompt_text = next((r['prompt'] for r in all_results[list(all_results.keys())[0]] 
                               if r['category'] == category), "")
            if prompt_text:
                f.write(f"Prompt: {prompt_text}\n\n")
            
            for model_name in all_results.keys():
                result = next((r for r in all_results[model_name] if r['category'] == category), None)
                if result:
                    f.write(f"Model: {model_name}\n")
                    if result.get('error', False):
                        f.write("Result: [ERROR]\n")
                    else:
                        # Write FULL completion without any truncation
                        f.write(f"Completion ({result['tokens_generated']} tokens in {result['generation_time']:.2f}s):\n")
                        f.write(result['completion'] + "\n")
                    f.write("\n")
            f.write("\n")
        
        # Write summary statistics
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 80 + "\n")
        for model_name, stats in full_results['summary'].items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  • Success Rate: {stats['success_rate']*100:.1f}%\n")
            f.write(f"  • Average Tokens: {stats['avg_tokens']:.1f}\n")
            f.write(f"  • Average Time: {stats['avg_time']:.2f}s\n")
            f.write(f"  • Total Tokens: {stats['total_tokens']}\n")
    
    print_message(f"Text summary saved to: {text_filename}", "info")


def display_results(results: Union[str, Dict[str, Dict[str, any]]], 
                   prompt: str, 
                   metadata: Optional[Dict[str, Dict]] = None,
                   mode: str = "comparison",
                   already_streamed: bool = False):
    """Unified display function for all result types.
    
    Args:
        results: Either a string (single model) or dict of results (multi-model)
        prompt: The prompt used
        metadata: Optional metadata
        mode: Display mode ("comparison", "general_knowledge")
        already_streamed: Whether the text was already displayed via streaming
    """
    # Single model result
    if isinstance(results, str):
        if not already_streamed:
            # Only display if not already streamed
            print(f"\n{Colors.GREEN}{Colors.BOLD}Generated Text:{Colors.ENDC}")
            print("=" * 60)
            print(results)
            print("=" * 60)
        return
    
    # Multi-model results
    title = "GENERAL KNOWLEDGE EVALUATION" if mode == "general_knowledge" else "MODEL COMPARISON RESULTS"
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    
    if mode != "general_knowledge":
        print(f"\n{Colors.BOLD}Prompt:{Colors.ENDC} {Colors.CYAN}{prompt}{Colors.ENDC}\n")
    
    # Display each model's output
    for idx, (model_name, result) in enumerate(results.items(), 1):
        print(f"\n{Colors.BLUE}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BLUE}{Colors.BOLD}Model {idx}: {model_name}{Colors.ENDC}")
        
        if metadata and model_name in metadata:
            model_metadata = metadata[model_name]
            print(f"{Colors.CYAN}  Parameters: {model_metadata['params']:,}, Memory: {model_metadata['memory_mb']:.1f} MB{Colors.ENDC}")
        
        print(f"{Colors.BLUE}{'─' * 70}{Colors.ENDC}")
        
        if result.get('error'):
            print(f"{Colors.RED}Generation failed: {result['text']}{Colors.ENDC}")
        else:
            text = result.get('text', result.get('completion', ''))
            if mode == "general_knowledge" and 'category' in result:
                print(f"Category: {result['category']}")
                print(f"Prompt: {result['prompt']}")
            
            # Truncate long text for display
            if len(text) > 200 and mode == "general_knowledge":
                text = text[:200] + "..."
            print(f"\n{text}\n")
            
            print(f"{Colors.CYAN}Statistics:{Colors.ENDC}")
            print(f"  • Tokens generated: {result.get('tokens_generated', 0)}")
            print(f"  • Generation time: {result.get('generation_time', 0):.2f}s")
            if result.get('tokens_per_sec'):
                print(f"  • Speed: {result['tokens_per_sec']:.1f} tokens/sec")
    
    # Performance summary
    valid_results = {k: v for k, v in results.items() if not v.get('error')}
    if valid_results:
        print(f"\n{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.GREEN}{Colors.BOLD}Performance Summary:{Colors.ENDC}")
        print(f"{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
        
        fastest = min(valid_results.items(), key=lambda x: x[1].get('generation_time', float('inf')))
        most_tokens = max(valid_results.items(), key=lambda x: x[1].get('tokens_generated', 0))
        
        print(f"  • Fastest: {fastest[0]} ({fastest[1]['generation_time']:.2f}s)")
        print(f"  • Most tokens: {most_tokens[0]} ({most_tokens[1]['tokens_generated']} tokens)")
        
        if mode == "general_knowledge":
            success_rate = len(valid_results) / len(results) * 100
            print(f"  • Success rate: {success_rate:.1f}%")
    
    print(f"{Colors.GREEN}{'=' * 70}{Colors.ENDC}")


def main():
    """Main inference loop."""
    print_message("Transformer Language Model Inference", "header")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print_message(f"Using GPU: {torch.cuda.get_device_name()}", "success")
        print_message(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", "info")
    else:
        device = torch.device('cpu')
        print_message("CUDA not available. Using CPU (will be slower)", "warning")
    
    # Select inference mode
    print(f"\n{Colors.BOLD}Select Inference Mode:{Colors.ENDC}")
    print("  1. Single model inference")
    print("  2. Multi-model comparison")
    print("  3. Benchmark models")
    print("  4. General knowledge evaluation")
    print("  0. Exit")
    
    while True:
        try:
            mode_choice = input(f"\n{Colors.BOLD}Select mode (0-4): {Colors.ENDC}")
            mode_choice = int(mode_choice)
            
            if mode_choice == 0:
                print_message("Exiting...", "info")
                sys.exit(0)
            elif mode_choice in [1, 2, 3, 4]:
                break
            else:
                print_message("Please enter 0, 1, 2, 3, or 4", "warning")
        except ValueError:
            print_message("Invalid input. Please enter a number.", "warning")
    
    # Select checkpoint directory
    base_checkpoint_dir = Path("checkpoints")
    if not base_checkpoint_dir.exists():
        print_message(f"Checkpoint directory not found: {base_checkpoint_dir}", "error")
        sys.exit(1)
    
    try:
        if mode_choice == 1:
            # Single model inference
            result = load_models_for_inference(base_checkpoint_dir, device, multi_model=False)
            if result is None:
                sys.exit(0)
            
            models_dict, metadata = result
            run_generation_loop(models_dict, device, metadata)
            
        elif mode_choice == 2:
            # Multi-model comparison
            result = load_models_for_inference(base_checkpoint_dir, device, multi_model=True)
            if result is None:
                sys.exit(0)
            
            models_dict, metadata = result
            run_generation_loop(models_dict, device, metadata)
        
        elif mode_choice == 3:
            # Benchmark mode - disable compilation to avoid Flash Attention issues
            result = load_models_for_inference(base_checkpoint_dir, device, multi_model=True, compile_model=False)
            if result is None:
                sys.exit(0)
            
            models_dict, metadata = result
            run_benchmark(models_dict, device)
        
        elif mode_choice == 4:
            # General knowledge evaluation
            print(f"\n{Colors.BOLD}Model Selection for General Knowledge Evaluation{Colors.ENDC}")
            print("  1. Single model evaluation")
            print("  2. Multi-model comparison")
            
            eval_choice = None
            while eval_choice not in [1, 2]:
                try:
                    eval_choice = int(input(f"\n{Colors.BOLD}Select (1-2): {Colors.ENDC}"))
                    if eval_choice not in [1, 2]:
                        print_message("Please enter 1 or 2", "warning")
                except ValueError:
                    print_message("Invalid input. Please enter a number.", "warning")
            
            result = load_models_for_inference(base_checkpoint_dir, device, multi_model=(eval_choice == 2))
            if result is None:
                sys.exit(0)
            
            models_dict, metadata = result
            run_general_knowledge_eval(models_dict, device, metadata)
    
    except KeyboardInterrupt:
        print_message("\nInterrupted by user", "warning")
    except Exception as e:
        print_message(f"Error: {e}", "error")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_session(device)


def cleanup_session(device: torch.device):
    """Clean up GPU cache and print session end message."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print_message("GPU cache cleared", "info")




def get_generation_params_interactive(defaults: Optional[Dict[str, any]] = None, include_streaming: bool = True) -> Tuple[Dict[str, any], bool]:
    """Get generation parameters interactively from user.
    
    Args:
        defaults: Optional dictionary of default values to use
        include_streaming: Whether to ask about streaming preference
    
    Returns:
        Tuple of (generation parameters dict, streaming enabled bool)
    """
    if defaults is None:
        defaults = {'max_length': 200, 'temperature': 1, 'top_k': 50, 'top_p': 0.95}
    
    print(f"\n{Colors.BOLD}Configure Generation:{Colors.ENDC}")
    print("  1. Use default parameters")
    print("  2. Customize parameters")
    
    param_choice = input(f"\n{Colors.BOLD}Select (1-2): {Colors.ENDC}")
    
    use_streaming = True  # Default to streaming
    
    if param_choice == '2':
        print(f"\n{Colors.BOLD}Generation Parameters:{Colors.ENDC}")
        print("  Press Enter to use default values")
        
        params = {}
        
        # Max length
        max_length_input = input(f"  Max length (default {defaults['max_length']}): ")
        params['max_length'] = int(max_length_input) if max_length_input else defaults['max_length']
        
        # Temperature 
        temp_input = input(f"  Temperature (default {defaults['temperature']}): ")
        params['temperature'] = float(temp_input) if temp_input else defaults['temperature']
        
        # Top-k
        top_k_input = input(f"  Top-k (default {defaults['top_k']}, 0 to disable): ")
        params['top_k'] = int(top_k_input) if top_k_input else defaults['top_k']
        params['top_k'] = params['top_k'] if params['top_k'] > 0 else None
        
        # Top-p
        top_p_input = input(f"  Top-p (default {defaults['top_p']}, 1.0 to disable): ")
        params['top_p'] = float(top_p_input) if top_p_input else defaults['top_p']
        params['top_p'] = params['top_p'] if params['top_p'] < 1.0 else None
        
        # Streaming option
        if include_streaming:
            stream_input = input(f"  Enable streaming (y/n, default y): ").lower()
            use_streaming = stream_input != 'n'
        
        return params, use_streaming
    else:
        print_message("Using default parameters", "info")
        return defaults, use_streaming














def run_generation_loop(models_dict: Dict[str, Tuple[TransformerLM, WikipediaTokenizer]], 
                       device: torch.device, 
                       metadata: Optional[Dict] = None):
    """Run text generation loop."""
    is_multi = len(models_dict) > 1
    session_type = "multi-model comparison" if is_multi else "single model generation"
    
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.CYAN}Starting {session_type} session...{Colors.ENDC}")
    if is_multi:
        print(f"{Colors.CYAN}{len(models_dict)} models loaded for comparison{Colors.ENDC}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    
    while True:
        print(f"\n{Colors.BOLD}Enter Prompt:{Colors.ENDC}")
        print("  Type your prompt or 'q' to quit")
        
        prompt = input(f"\n{Colors.BOLD}> {Colors.ENDC}")
        
        if prompt.lower() in ['q', 'quit']:
            print_message("Exiting...", "info")
            break
        
        if not prompt.strip():
            print_message("Empty prompt. Please try again.", "warning")
            continue
        
        # Get generation parameters
        defaults = {'max_length': 512 if is_multi else 200, 'temperature': 1, 'top_k': 50, 'top_p': 0.95}
        generation_params, use_streaming = get_generation_params_interactive(defaults)
        
        print(f"\n{Colors.BOLD}Generating...{Colors.ENDC}")
        print("-" * 60)
        
        try:
            # Use streaming based on user preference
            results = generate_text_unified(models_dict, prompt, device, stream=use_streaming, **generation_params)
            # For single model, the full text is already displayed via streaming
            if isinstance(results, str):
                # Single model - text was already streamed, just show separator
                print("=" * 60)
            else:
                # Multi-model - show comparison results
                display_results(results, prompt, metadata)
        except Exception as e:
            print_message(f"Generation failed: {e}", "error")
            import traceback
            traceback.print_exc()
        
        # Continue prompt
        cont = input("\nPress Enter to generate more, or 'q' to quit: ")
        if cont.lower() == 'q':
            break


def run_benchmark(models_dict: Dict[str, Tuple[TransformerLM, WikipediaTokenizer]], device: torch.device):
    """Run benchmark evaluation."""
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.CYAN}Starting benchmark evaluation...{Colors.ENDC}")
    print(f"{Colors.CYAN}{len(models_dict)} models loaded for benchmarking{Colors.ENDC}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.ENDC}")
    
    configs = [
        ('Quick', {
            'lambada_samples': 50
        }),
        ('Standard', {
            'lambada_samples': 100
        }),
        ('Comprehensive', {
            'lambada_samples': 200
        })
    ]
    
    config_items = [(f"{name} ({cfg['lambada_samples']} LAMBADA samples)", cfg) for name, cfg in configs]
    config = select_items_interactive(config_items, "Select LAMBADA Benchmark Configuration:")
    
    if config is None:
        print_message("Benchmark cancelled.", "info")
        return
    
    print(f"\n{Colors.BOLD}Running Benchmarks...{Colors.ENDC}")
    print_message("This may take several minutes.", "warning")
    
    benchmark = ModelBenchmark(device)
    results = benchmark.run_full_benchmark(models_dict, benchmark_config=config)
    
    print("\n" + format_benchmark_results(results))
    save_benchmark_results(results)
    print_message("\nBenchmark completed successfully!", "success")


def run_general_knowledge_eval(models_dict: Dict[str, Tuple[TransformerLM, WikipediaTokenizer]], 
                              device: torch.device, 
                              metadata: Optional[Dict] = None):
    """Run general knowledge evaluation."""
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.CYAN}Starting General Knowledge Evaluation...{Colors.ENDC}")
    print(f"{Colors.CYAN}{len(models_dict)} model(s) loaded for evaluation{Colors.ENDC}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.ENDC}")
    
    prompts = get_general_knowledge_prompts()
    print(f"\n{Colors.BOLD}Evaluating {len(prompts)} knowledge areas{Colors.ENDC}")
    
    eval_defaults = {'max_length': 150, 'temperature': 1, 'top_k': 40, 'top_p': 0.9}
    # For evaluation, disable streaming to avoid interfering with progress display
    generation_params, _ = get_generation_params_interactive(eval_defaults, include_streaming=False)
    
    all_results = {}
    
    # Initialize results structure for all models
    for model_name in models_dict.keys():
        all_results[model_name] = []
    
    for prompt_idx, (category, prompt) in enumerate(prompts, 1):
        print(f"\n{Colors.BLUE}[{prompt_idx}/{len(prompts)}] {category}{Colors.ENDC}")
        
        # Always call generate_text_unified with all models to get consistent dict results
        # This works for both single and multiple models
        try:
            # Disable streaming for evaluation to avoid interfering with progress display
            results = generate_text_unified(models_dict, prompt, device, stream=False, **generation_params)
            
            # Handle the return value based on whether it's a string (single model) or dict (multiple models)
            if isinstance(results, str):
                # Single model case - results is just the generated text
                model_name = list(models_dict.keys())[0]
                # Get the tokenizer from the model tuple
                _, tokenizer = models_dict[model_name]
                
                generated_text = results
                completion = generated_text[len(prompt):].strip() if len(generated_text) > len(prompt) else ""
                
                # Calculate tokens properly using the actual tokenizer
                if completion:
                    tokens_generated = len(tokenizer.encode(completion, add_special_tokens=False))
                else:
                    tokens_generated = 0
                
                # Note: We can't get exact generation time from string return, but the stats
                # were already printed to console by generate_text_unified
                all_results[model_name].append({
                    'category': category,
                    'prompt': prompt,
                    'completion': completion,
                    'generation_time': 1.0,  # Placeholder since exact time was printed but not returned
                    'tokens_generated': tokens_generated
                })
            else:
                # Multiple models case - results is a dict
                for model_name in models_dict.keys():
                    if model_name in results and not results[model_name].get('error'):
                        result = results[model_name]
                        all_results[model_name].append({
                            'category': category,
                            'prompt': prompt,
                            'completion': result['text'][len(prompt):].strip(),
                            'generation_time': result['generation_time'],
                            'tokens_generated': result['tokens_generated']
                        })
                    else:
                        # Handle error case
                        error_msg = results.get(model_name, {}).get('text', '[Generation failed]')
                        all_results[model_name].append({
                            'category': category,
                            'prompt': prompt,
                            'completion': error_msg,
                            'generation_time': 0,
                            'tokens_generated': 0,
                            'error': True
                        })
        except Exception as e:
            print_message(f"Generation failed: {e}", "error")
            for model_name in models_dict.keys():
                all_results[model_name].append({
                    'category': category,
                    'prompt': prompt,
                    'completion': f"[ERROR: {str(e)}]",
                    'generation_time': 0,
                    'tokens_generated': 0,
                    'error': True
                })
    
    # Display comprehensive results summary
    print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}GENERAL KNOWLEDGE EVALUATION RESULTS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    
    # Calculate and display summary statistics for each model
    print(f"\n{Colors.GREEN}{Colors.BOLD}Model Performance Summary:{Colors.ENDC}")
    print(f"{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
    
    for model_name, results_list in all_results.items():
        successful_results = [r for r in results_list if not r.get('error', False)]
        
        if successful_results:
            avg_time = sum(r['generation_time'] for r in successful_results) / len(successful_results)
            total_tokens = sum(r['tokens_generated'] for r in successful_results)
            avg_tokens = total_tokens / len(successful_results)
            success_rate = (len(successful_results) / len(results_list)) * 100
            
            print(f"\n{Colors.BLUE}{Colors.BOLD}Model: {model_name}{Colors.ENDC}")
            if metadata and model_name in metadata:
                model_meta = metadata[model_name]
                print(f"  • Parameters: {model_meta['params']:,}")
                print(f"  • Memory: {model_meta['memory_mb']:.1f} MB")
            print(f"  • Success Rate: {success_rate:.1f}% ({len(successful_results)}/{len(results_list)} prompts)")
            print(f"  • Average Generation Time: {avg_time:.2f}s")
            print(f"  • Average Tokens Generated: {avg_tokens:.1f}")
            print(f"  • Total Tokens Generated: {total_tokens}")
            
            # Calculate average tokens per second
            total_time = sum(r['generation_time'] for r in successful_results)
            if total_time > 0:
                avg_speed = total_tokens / total_time
                print(f"  • Average Speed: {avg_speed:.1f} tokens/sec")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}Model: {model_name}{Colors.ENDC}")
            print(f"  • All evaluations failed")
    
    # Display detailed results by category
    print(f"\n{Colors.GREEN}{Colors.BOLD}Detailed Results by Category:{Colors.ENDC}")
    print(f"{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
    
    categories_seen = []
    for model_name, results_list in all_results.items():
        for result in results_list:
            if result['category'] not in categories_seen:
                categories_seen.append(result['category'])
    
    for category in categories_seen:
        print(f"\n{Colors.CYAN}{Colors.BOLD}{category}:{Colors.ENDC}")
        
        # Find the prompt for this category
        prompt_text = next((r['prompt'] for r in all_results[list(all_results.keys())[0]] 
                           if r['category'] == category), "")
        if prompt_text:
            print(f"  Prompt: \"{prompt_text[:80]}...\"" if len(prompt_text) > 80 else f"  Prompt: \"{prompt_text}\"")
        
        for model_name in all_results.keys():
            result = next((r for r in all_results[model_name] if r['category'] == category), None)
            if result:
                if result.get('error', False):
                    print(f"\n  {Colors.RED}{model_name}: [ERROR]{Colors.ENDC}")
                else:
                    completion = result['completion']
                    # Truncate long completions for display (increased from 150 to 500 for better visibility)
                    if len(completion) > 500:
                        completion = completion[:500] + "..."
                    print(f"\n  {Colors.BLUE}{model_name}:{Colors.ENDC}")
                    print(f"    {completion}")
                    print(f"    {Colors.CYAN}({result['tokens_generated']} tokens in {result['generation_time']:.2f}s){Colors.ENDC}")
    
    # Display overall comparison if multiple models
    if len(all_results) > 1:
        print(f"\n{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.GREEN}{Colors.BOLD}Overall Comparison:{Colors.ENDC}")
        print(f"{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
        
        # Find best performing model
        model_scores = {}
        for model_name, results_list in all_results.items():
            successful = [r for r in results_list if not r.get('error', False)]
            if successful:
                avg_time = sum(r['generation_time'] for r in successful) / len(successful)
                success_rate = len(successful) / len(results_list)
                avg_tokens = sum(r['tokens_generated'] for r in successful) / len(successful)
                # Combined score (lower is better for time, higher for success rate and tokens)
                model_scores[model_name] = {
                    'success_rate': success_rate,
                    'avg_time': avg_time,
                    'avg_tokens': avg_tokens
                }
        
        if model_scores:
            # Best success rate
            best_success = max(model_scores.items(), key=lambda x: x[1]['success_rate'])
            print(f"  • Highest Success Rate: {best_success[0]} ({best_success[1]['success_rate']*100:.1f}%)")
            
            # Fastest generation
            fastest = min(model_scores.items(), key=lambda x: x[1]['avg_time'])
            print(f"  • Fastest Generation: {fastest[0]} ({fastest[1]['avg_time']:.2f}s avg)")
            
            # Most productive (tokens)
            most_productive = max(model_scores.items(), key=lambda x: x[1]['avg_tokens'])
            print(f"  • Most Productive: {most_productive[0]} ({most_productive[1]['avg_tokens']:.1f} tokens avg)")
    
    print(f"\n{Colors.GREEN}{'=' * 70}{Colors.ENDC}")
    
    # Save results to file
    print_message("\nSaving results to file...", "info")
    save_general_knowledge_results(all_results, metadata)
    
    # Offer to display full untruncated results
    print(f"\n{Colors.BOLD}Display Options:{Colors.ENDC}")
    print("  1. Show full untruncated results (may be long)")
    print("  2. Skip to completion")
    
    display_choice = input(f"\n{Colors.BOLD}Select (1-2): {Colors.ENDC}")
    
    if display_choice == '1':
        print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}FULL UNTRUNCATED RESULTS{Colors.ENDC}")
        print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
        
        for category in categories_seen:
            print(f"\n{Colors.CYAN}{Colors.BOLD}CATEGORY: {category}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
            
            # Find the prompt for this category
            prompt_text = next((r['prompt'] for r in all_results[list(all_results.keys())[0]] 
                               if r['category'] == category), "")
            if prompt_text:
                print(f"Prompt: {prompt_text}\n")
            
            for model_name in all_results.keys():
                result = next((r for r in all_results[model_name] if r['category'] == category), None)
                if result:
                    print(f"\n{Colors.BLUE}{Colors.BOLD}Model: {model_name}{Colors.ENDC}")
                    if result.get('error', False):
                        print(f"{Colors.RED}[ERROR]{Colors.ENDC}")
                    else:
                        # Display FULL completion without truncation
                        print(f"Full Completion ({result['tokens_generated']} tokens, {result['generation_time']:.2f}s):")
                        print(f"{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
                        print(result['completion'])
                        print(f"{Colors.GREEN}{'─' * 70}{Colors.ENDC}")
        
        print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
        print_message("Full results displayed above. Also saved to file.", "info")
    
    print_message("\nGeneral knowledge evaluation completed!", "success")


if __name__ == "__main__":
    main()
