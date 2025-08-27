#!/usr/bin/env python3
"""
Main router for the LLM training pipeline.
Supports both interactive menu mode and direct CLI execution.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Module mappings
MODULES = {
    "cleanup": {
        "script": "cleaner.py",
        "description": "Clean and process raw Wikipedia data",
        "args_help": "Options: --test (process limited articles for testing)"
    },
    "tokenize": {
        "script": "tokenizer.py", 
        "description": "Manage CodeLlama tokenizer operations",
        "args_help": "Options: --test, --encode TEXT, --decode IDS, --save PATH, --load PATH"
    },
    "prepare": {
        "script": "dataset_prep.py",
        "description": "Prepare tokenized datasets for training",
        "args_help": "Options: --test (process subset), --window SIZE (default: 512)"
    },
    "train": {
        "script": "train.py",
        "description": "Train the transformer language model",
        "args_help": "Options: --test, --config PATH, --resume CHECKPOINT, --name EXPERIMENT"
    },
    "inference": {
        "script": "Inference.py",
        "description": "Run interactive text generation with trained models",
        "args_help": "(Interactive mode only - no command line arguments)"
    }
}


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print(" " * 15 + "LLM Training Pipeline Router")
    print("="*60)


def print_menu():
    """Display interactive menu."""
    print("\nAvailable modules:")
    print("-" * 40)
    print("1) Raw Cleanup       - Process Wikipedia dumps")
    print("2) Tokenization      - Manage tokenizer operations")
    print("3) Dataset Prep      - Prepare training datasets")
    print("4) Training          - Train language model")
    print("5) Inference         - Generate text interactively")
    print("6) Quit")
    print("-" * 40)


def get_module_by_number(number):
    """Map menu number to module name."""
    mapping = {
        1: "cleanup",
        2: "tokenize",
        3: "prepare",
        4: "train",
        5: "inference"
    }
    return mapping.get(number)


def run_module(module_name, args):
    """Execute a module with given arguments."""
    module_info = MODULES.get(module_name)
    if not module_info:
        print(f"Error: Unknown module '{module_name}'")
        return 1
    
    script_path = Path(__file__).parent / module_info["script"]
    if not script_path.exists():
        print(f"Error: Module script '{script_path}' not found")
        return 1
    
    # Build command
    cmd = [sys.executable, str(script_path)] + args
    
    # Display what we're running
    print(f"\n{'='*60}")
    print(f"Executing: {module_info['description']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # Run the module
    try:
        result = subprocess.run(cmd, check=False)
        
        # Check for errors from submodule
        if result.returncode != 0:
            if result.returncode == 130:  # Ctrl-C in submodule
                print("\n\nModule execution interrupted by user")
            elif result.returncode > 0:  # Non-zero exit indicates an error
                print(f"\nModule exited with code {result.returncode}")
                # For critical failures (not user interrupts), raise an exception
                # This allows the main menu to decide whether to continue
                if result.returncode not in [130]:  # 130 is Ctrl-C, which is OK
                    raise RuntimeError(f"Module '{module_name}' failed with exit code {result.returncode}")
        
        return result.returncode
    except KeyboardInterrupt:
        # This happens if Ctrl-C is pressed while subprocess is starting
        print("\n\nModule execution interrupted by user")
        raise  # Re-raise to exit properly
    except RuntimeError:
        # Re-raise RuntimeError as-is (from our check above)
        raise
    except Exception as e:
        print(f"Error running module: {e}")
        raise  # Re-raise to propagate the error


def interactive_mode():
    """Run in interactive menu mode."""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("\nSelect module (1-6): ").strip()
            
            if not choice:
                continue
                
            if choice == "6":
                print("\nExiting pipeline router. Goodbye!")
                return 0
            
            try:
                choice_num = int(choice)
                if choice_num < 1 or choice_num > 6:
                    print("Invalid choice. Please select 1-6.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number 1-6.")
                continue
            
            module_name = get_module_by_number(choice_num)
            if not module_name:
                print("Invalid module selection.")
                continue
            
            module_info = MODULES[module_name]
            
            # Get additional arguments if needed
            args = []
            if module_name != "inference":  # Inference has no CLI args
                print(f"\n{module_info['args_help']}")
                args_input = input("Enter arguments (or press Enter for defaults): ").strip()
                
                if args_input:
                    # Parse the arguments - handle quoted strings
                    import shlex
                    args = shlex.split(args_input)
            
            # Run the module
            return_code = run_module(module_name, args)
            
            if return_code != 0:
                print(f"\nModule exited with code {return_code}")
            
            # Ask if user wants to continue
            continue_choice = input("\nRun another module? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\nExiting pipeline router. Goodbye!")
                return 0
                
        except (KeyboardInterrupt, EOFError):
            # Let these exceptions bubble up to main() for proper handling
            raise
        except RuntimeError as e:
            # Critical errors from submodules
            print(f"\nCritical error: {e}")
            error_choice = input("\nContinue despite error? (y/n): ").strip().lower()
            if error_choice != 'y':
                raise  # Re-raise to exit with error


def cli_mode():
    """Run in CLI mode with argparse."""
    parser = argparse.ArgumentParser(
        description="LLM Training Pipeline Router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode
  python main.py cleanup --test           # Run cleanup in test mode
  python main.py tokenize --encode "text" # Encode text
  python main.py prepare --window 1024    # Prepare with custom window
  python main.py train --test --name exp1 # Train with experiment name
  python main.py inference                # Run inference (interactive)
        """
    )
    
    # Create subparsers for each module
    subparsers = parser.add_subparsers(dest='module', help='Module to run')
    
    # Cleanup module parser
    cleanup_parser = subparsers.add_parser(
        'cleanup',
        help='Clean and process raw Wikipedia data'
    )
    cleanup_parser.add_argument('--test', action='store_true',
                                help='Process limited articles for testing')
    
    # Tokenize module parser
    tokenize_parser = subparsers.add_parser(
        'tokenize',
        help='Manage CodeLlama tokenizer operations'
    )
    tokenize_parser.add_argument('--test', action='store_true',
                                 help='Run tokenizer tests')
    tokenize_parser.add_argument('--encode', type=str,
                                 help='Text to encode')
    tokenize_parser.add_argument('--decode', type=str,
                                 help='Token IDs to decode (comma-separated)')
    tokenize_parser.add_argument('--save', type=str,
                                 help='Path to save tokenizer')
    tokenize_parser.add_argument('--load', type=str,
                                 help='Path to load tokenizer from')
    
    # Prepare module parser
    prepare_parser = subparsers.add_parser(
        'prepare',
        help='Prepare tokenized datasets for training'
    )
    prepare_parser.add_argument('--test', action='store_true',
                                help='Process subset for testing')
    prepare_parser.add_argument('--window', type=int, default=512,
                                help='Window size for sequences (default: 512)')
    
    # Train module parser
    train_parser = subparsers.add_parser(
        'train',
        help='Train the transformer language model'
    )
    train_parser.add_argument('--test', action='store_true',
                             help='Run single epoch test')
    train_parser.add_argument('--config', type=str,
                             help='Path to config file')
    train_parser.add_argument('--resume', type=str,
                             help='Path to checkpoint to resume from')
    train_parser.add_argument('--name', type=str,
                             help='Experiment name for logging')
    
    # Inference module parser
    inference_parser = subparsers.add_parser(
        'inference',
        help='Run interactive text generation with trained models'
    )
    # No arguments for inference - it's interactive only
    
    args = parser.parse_args()
    
    # If no module specified, run interactive mode
    if not args.module:
        return interactive_mode()
    
    # Build argument list for the module
    module_args = []
    if args.module == 'cleanup':
        if args.test:
            module_args.append('--test')
            
    elif args.module == 'tokenize':
        if args.test:
            module_args.append('--test')
        if args.encode:
            module_args.extend(['--encode', args.encode])
        if args.decode:
            module_args.extend(['--decode', args.decode])
        if args.save:
            module_args.extend(['--save', args.save])
        if args.load:
            module_args.extend(['--load', args.load])
            
    elif args.module == 'prepare':
        if args.test:
            module_args.append('--test')
        if args.window != 512:  # Only add if not default
            module_args.extend(['--window', str(args.window)])
            
    elif args.module == 'train':
        if args.test:
            module_args.append('--test')
        if args.config:
            module_args.extend(['--config', args.config])
        if args.resume:
            module_args.extend(['--resume', args.resume])
        if args.name:
            module_args.extend(['--name', args.name])
            
    # elif args.module == 'inference':
    #     No arguments needed - runs interactively
    
    # Run the selected module
    return run_module(args.module, module_args)


def main():
    """Main entry point."""
    try:
        # If no arguments provided, run interactive mode
        if len(sys.argv) == 1:
            return interactive_mode()
        else:
            return cli_mode()
            
    except KeyboardInterrupt:
        print("\n\nPipeline router interrupted. Goodbye!")
        return 130
    except EOFError:
        print("\n\nExiting pipeline router. Goodbye!")
        return 0
    except RuntimeError as e:
        print(f"\nCritical error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())