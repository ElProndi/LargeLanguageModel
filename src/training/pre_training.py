#!/usr/bin/env python3
"""Main training script for transformer language model."""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Import project modules
try:
    # Try relative imports (when imported as a module)
    from ..utils.model import create_model
    from .dataloader import create_fixed_eval_dataloaders, destroy_dataloaders, calculate_fixed_eval_dataloader_stats
    from ..utils.logging_utils import DualLogger
    from ..utils.scheduler import get_cosine_schedule_with_warmup
    from ..utils.metrics import MetricsTracker
    from ..dataset_preparation.tokenizer import CodeLlamaTokenizer
except ImportError:
    # Fall back to absolute imports (when run directly)
    from src.utils.model import create_model
    from src.training.dataloader import create_fixed_eval_dataloaders, destroy_dataloaders, calculate_fixed_eval_dataloader_stats
    from src.utils.logging_utils import DualLogger
    from src.utils.scheduler import get_cosine_schedule_with_warmup
    from src.utils.metrics import MetricsTracker
    from src.dataset_preparation.tokenizer import CodeLlamaTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


class Trainer:
    """Main trainer class for language model training."""
    
    def __init__(
        self,
        config_path: str = "config.json",
        model_size: str = None,
        test_mode: bool = False,
        resume_from: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration file
            model_size: Model size to use ("small" or "medium"). If None, uses default from config.
            test_mode: Train on 1 chunk (subset) out of 64 total chunks
            resume_from: Path to checkpoint to resume from
            experiment_name: Name for this training run
        """
        print(f"Initializing Transformer Language Model Trainer")
        
        # Store model size
        self.model_size = model_size
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Display selected model size
        if 'models' in self.config:
            actual_model_size = model_size or self.config.get('default_model_size', 'medium')
            print(f"  • Model size: {actual_model_size}")
        
        self.test_mode = test_mode
        if test_mode:
            print("🧪 Running in TEST MODE - training on 1 chunk out of 64 total chunks")
            # max_steps will be calculated dynamically based on test dataset size
            self.config['logging']['eval_steps'] = 50  # Not used anymore, validation_interval is used instead
            self.config['logging']['save_steps'] = 50   # Save checkpoint every 50 steps in test mode
            self.config['logging']['logging_steps'] = 5
        
        # Setup device - CUDA is required
        if not torch.cuda.is_available():
            print("❌ CUDA is not available. GPU is required for training.")
            sys.exit(1)
        
        self.device = torch.device('cuda')  # Always use CUDA
        print(f"Device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Generate experiment name for both logging and checkpoints
        if experiment_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"run_{timestamp}"
        self.experiment_name = experiment_name
        
        # Initialize logging
        self.logger = DualLogger(
            log_dir=self.config['paths'].get('log_dir', 'logs'),
            experiment_name=self.experiment_name
        )
        self.logger.log_config(self.config)
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker(window_size=100)
        
        # Setup model
        print("\nSetting up model...")
        self.model = self._setup_model()
        
        # Setup tokenizer for visualization during validation
        print("Loading tokenizer for validation visualization...")
        self.tokenizer = CodeLlamaTokenizer()
        self.tokenizer.load(self.config['paths']['tokenizer_dir'])
        print(f"Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")
        
        # Initialize dataloader variables (will be created dynamically)
        self.train_loader = None
        self.eval_loader = None  # Fixed eval dataset loader (loaded once)
        self.train_dataset = None
        self.eval_dataset = None  # Fixed eval dataset
        self.current_data_subset = None
        self.dataloader_info = None
        
        # Initialize training state early for calculations
        self.global_step = 0
        # max_steps will be calculated dynamically based on dataset size
        self.best_val_loss = float('inf')
        self.completed_subsets = []  # Track which subset indices have been completed
        self.recent_step_times = deque(maxlen=5)  # Track last 5 step times for ETA calculation
        
        # Get gradient accumulation steps early for correct calculations
        self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        # Get dataset source from config
        self.dataset_source = self.config.get('dataset', {}).get('source', 'fineweb')
        print(f"Using dataset: {self.dataset_source}")
        
        # Calculate training steps without loading data (optimized)
        # This avoids loading ~13GB into GPU memory just to count batches
        print("Calculating training steps...", end='')
        
        # Configure number of data subsets (default to 32)
        self.num_data_subsets = self.config['training'].get('num_data_subsets', 64)
        
        # Get statistics for all subsets using fixed eval mode
        subset_stats = []
        for subset_idx in range(self.num_data_subsets):
            stats = calculate_fixed_eval_dataloader_stats(
                batch_size=self.config['training']['batch_size'],
                test_mode=self.test_mode,
                subset_index=subset_idx,
                num_subsets=self.num_data_subsets
            )
            subset_stats.append(stats)
        
        # Calculate batch counts and dynamic max_steps based on actual dataset size
        num_epochs = self.config['training'].get('num_epochs', 1)
        
        if self.test_mode:
            # Test mode - calculate steps for first subset only (1/32 of the data)
            self.batches_per_epoch = subset_stats[0]['train_batches']
            self.steps_per_epoch = self.batches_per_epoch // self.gradient_accumulation_steps
            if self.batches_per_epoch % self.gradient_accumulation_steps != 0:
                self.steps_per_epoch += 1
            
            # Set max_steps for test mode with epochs
            self.max_steps = self.steps_per_epoch * num_epochs
            self.total_training_steps = self.max_steps
            
            if not resume_from:
                print(f"  • Training on subset 1/{self.num_data_subsets} only (1 chunk out of {self.num_data_subsets} total): {self.steps_per_epoch} steps/epoch × {num_epochs} epochs = {self.max_steps} total steps")
        else:
            # Train on all subsets sequentially - sum all subsets
            self.batches_per_epoch = sum(stats['train_batches'] for stats in subset_stats)
            
            # With gradient accumulation, optimizer steps = batches / accumulation_steps
            self.steps_per_epoch = self.batches_per_epoch // self.gradient_accumulation_steps
            # Handle remainder batches (last incomplete accumulation in epoch)
            if self.batches_per_epoch % self.gradient_accumulation_steps != 0:
                self.steps_per_epoch += 1
            
            # Set max_steps for all subsets with epochs
            self.max_steps = self.steps_per_epoch * num_epochs
            self.total_training_steps = self.max_steps
            
            if not resume_from:
                print(f"  • Training on all {self.num_data_subsets} subsets: {self.steps_per_epoch} steps/epoch × {num_epochs} epochs = {self.max_steps} total steps")
        
        # Validation frequency: doubled - every 500 steps (or less for test mode)
        if self.test_mode:
            self.validation_interval = 25  # Validate every 25 steps in test mode
        else:
            self.validation_interval = max(500, self.max_steps // 20)  # At least every 500 steps
        
        # Setup training components (scheduler needs total_training_steps)
        print(" done.\nSetting up optimizer and scheduler...")
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup checkpointing with run-specific subdirectory
        base_checkpoint_dir = Path(self.config['paths'].get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir = base_checkpoint_dir / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"\nTraining configuration:")
        print(f"  • Batch size: {self.config['training']['batch_size']}")
        print(f"  • Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"  • Effective batch size: {self.config['training']['batch_size'] * self.gradient_accumulation_steps}")
        print(f"  • Learning rate: {self.config['training']['learning_rate']}")
        print(f"  • Maximum training steps: {self.max_steps:,}")
        print(f"  • Batches per epoch: {self.batches_per_epoch}")
        print(f"  • Optimizer steps per epoch: {self.steps_per_epoch}")
        print(f"  • Total training steps: {self.total_training_steps:,}")
        print(f"  • Warmup steps: {int(self.max_steps * self.config['training']['warmup_ratio'])} (1% of max steps)")
        print(f"  • Cosine target: {int(self.max_steps * self.config['training']['cosine_target_ratio'])} steps (80% of training)")
        print(f"  • Validation interval: every {self.validation_interval} steps")
        print(f"  • Checkpoint saving: every {self.config['logging']['save_steps']} steps + after each validation")
    
    def _setup_model(self) -> nn.Module:
        """Setup and optionally compile the model.
        
        Returns:
            Configured model (already on GPU from model.py)
        """
        # Create model with specified size - automatically moved to GPU in model.py
        model = create_model(
            config_path=self.config.get('config_path', 'config.json'),
            model_size=self.model_size
        )
        # No need to call model.to(device) - model is already on GPU
        
        # Model compilation (commented out by default)
        # Uncomment to enable torch.compile for faster training:

        #print("Compiling model with torch.compile()...")
        #model = torch.compile(model)
        
        return model
    
    def _create_dataloaders(self, subset_index: int = 0, num_subsets: int = None) -> Tuple[DataLoader, DataLoader, Dict]:
        """Create train and eval data loaders using fixed evaluation dataset.
        
        The eval dataset (tokens_0.npy) is loaded once on first call and reused.
        Training datasets are loaded from tokens_1.npy onwards based on subset_index.
        
        Args:
            subset_index: 0-based index of the subset to load for training
            num_subsets: Total number of subsets (defaults to self.num_data_subsets)
        
        Returns:
            Tuple of (train_loader, eval_loader, info_dict)
        """
        if num_subsets is None:
            num_subsets = self.num_data_subsets
        
        subset_desc = f"subset {subset_index+1}/{num_subsets}"
        print(f"\nLoading data: {subset_desc}")
        
        # Create dataloaders with fixed eval dataset
        train_loader, eval_loader, info = create_fixed_eval_dataloaders(
            batch_size=self.config['training']['batch_size'],
            test_mode=self.test_mode,
            verbose=True,
            subset_index=subset_index,  # Load only specified subset for training
            num_subsets=num_subsets  # Total number of subsets
        )
        
        # Store references
        self.train_loader = train_loader
        
        # Only store eval loader on first load (it's the same for all subsets)
        if self.eval_loader is None:
            self.eval_loader = eval_loader
            self.eval_dataset = info['eval_dataset']
            print(f"Fixed eval dataset stored: {info['eval_size']:,} sequences")
        
        self.train_dataset = info['train_dataset']
        self.current_data_subset_index = subset_index
        self.dataloader_info = info
        
        return train_loader, self.eval_loader, info
    
    def _switch_data_subset(self, new_subset_index: int):
        """Switch to a different training data subset, keeping eval dataset intact.
        
        Args:
            new_subset_index: 0-based index of the subset to switch to
        """
        if hasattr(self, 'current_data_subset_index') and self.current_data_subset_index == new_subset_index:
            print(f"Already using subset {new_subset_index+1}/{self.num_data_subsets}, skipping switch")
            return
        
        old_desc = f"subset {self.current_data_subset_index+1}/{self.num_data_subsets}" if hasattr(self, 'current_data_subset_index') else "unknown"
        new_desc = f"subset {new_subset_index+1}/{self.num_data_subsets}"
        print(f"\nSwitching training data: {old_desc} → {new_desc}")
        
        # Destroy current training dataloader if it exists (keep eval loader)
        if self.train_loader is not None:
            print("Destroying old training dataloader...")
            
            # Delete training dataset and dataloader only
            if hasattr(self.train_dataset, 'data'):
                del self.train_dataset.data
            del self.train_dataset
            del self.train_loader
            
            # Note: We keep eval_loader and eval_dataset intact
            
            # Clear partial info (keeping eval-related info)
            if self.dataloader_info is not None:
                # Remove train-specific info but keep eval info
                self.dataloader_info['train_dataset'] = None
                self.dataloader_info['train_size'] = 0
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        # Create new training dataloader for the new subset (eval stays the same)
        self._create_dataloaders(new_subset_index)
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup AdamW optimizer with weight decay.
        
        Returns:
            Configured optimizer
        """
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Don't apply weight decay to bias and layer norm parameters
            if 'bias' in name or 'layer_norm' in name or 'layernorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config['training']['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(
            optimizer_groups,
            lr=self.config['training']['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True
        )
        
        print(f"Optimizer: AdamW configured")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler.
        
        Returns:
            Configured scheduler
        """
        # Calculate warmup steps from ratio
        warmup_steps = int(self.max_steps * self.config['training']['warmup_ratio'])
        
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_training_steps,
            min_lr_ratio=self.config['training']['min_learning_rate'] / self.config['training']['learning_rate'],
            cosine_target_ratio=self.config['training']['cosine_target_ratio']
        )
        
        return scheduler
    
    def train_step(self, batch: torch.Tensor, accumulation_steps: int) -> float:
        """Perform single forward and backward pass.
        
        Args:
            batch: Input token IDs
            accumulation_steps: Number of gradient accumulation steps
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Batch is already on GPU from the dataloader
        
        # Always use bfloat16 autocast for forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model(input_ids=batch, labels=batch)
            loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        # This ensures gradients are averaged across accumulation steps
        scaled_loss = loss / accumulation_steps
        
        # Backward pass (accumulates gradients)
        scaled_loss.backward()
        
        # Return the unscaled loss for logging
        return loss.item()
    
    def optimizer_step(self) -> float:
        """Perform optimizer step with gradient clipping.
        
        Returns:
            Gradient norm after clipping
        """
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['max_grad_norm']
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step()
        
        # Zero gradients for next accumulation
        self.optimizer.zero_grad(set_to_none=True)
        
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop using fixed evaluation dataset.
        
        Evaluation data is transferred from CPU to GPU on-demand,
        then cleaned up after validation to save VRAM.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        num_batches = 0
        
        # Use subset of evaluation data for speed
        max_val_batches = 50 if not self.test_mode else 10
        
        # Flag to track if we've shown visualization
        visualization_shown = False
        
        for i, batch in enumerate(self.eval_loader):
            if i >= max_val_batches:
                break
            
            # Transfer batch from CPU to GPU (eval data is on CPU)
            # No non_blocking since we need immediate computation
            batch = batch.to(self.device)
            
            # Forward pass with bfloat16 autocast (same as training)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(input_ids=batch, labels=batch)
                loss = outputs['loss']
                
                # Store loss value while in autocast context
                loss_value = loss.item()
            
            # Visualization: Show model predictions for first batch only
            if i == 0 and not visualization_shown:
                visualization_shown = True
                print("\n" + "="*80)
                print("📊 MODEL PREDICTION VISUALIZATION (First validation sequence)")
                print("="*80)
                
                # Get the first sequence from the batch
                input_sequence = batch[0]  # Shape: [2048]
                
                # Get model predictions
                logits = outputs['logits'][0]  # Shape: [2048, vocab_size]
                predicted_tokens = torch.argmax(logits, dim=-1)  # Shape: [2048]
                
                # For language modeling, predictions are for the NEXT token
                # So predicted_tokens[i] is the prediction for what comes after input_sequence[i]
                
                # Decode tokens to text
                # Input: positions 0-2047
                # Predictions: what comes after positions 0-2047 (so predictions for positions 1-2048)
                input_text = self.tokenizer.decode(input_sequence.cpu().tolist(), skip_special_tokens=False)
                
                # For display, we'll show:
                # - Input tokens 0-2046 (all but last)
                # - Target tokens 1-2047 (what should come next)
                # - Predicted tokens for positions 1-511
                input_display = input_sequence[:-1].cpu().tolist()  # Tokens 0-510
                target_display = input_sequence[1:].cpu().tolist()  # Tokens 1-511 (targets)
                predicted_display = predicted_tokens[:-1].cpu().tolist()  # Predictions for 1-511
                
                # Decode for display
                input_text_display = self.tokenizer.decode(input_display, skip_special_tokens=False)
                target_text_display = self.tokenizer.decode(target_display, skip_special_tokens=False)
                predicted_text_display = self.tokenizer.decode(predicted_display, skip_special_tokens=False)
                
                # Truncate for terminal display
                max_display_chars = 300
                if len(input_text_display) > max_display_chars:
                    input_text_display = input_text_display[:max_display_chars] + "..."
                if len(target_text_display) > max_display_chars:
                    target_text_display = target_text_display[:max_display_chars] + "..."
                if len(predicted_text_display) > max_display_chars:
                    predicted_text_display = predicted_text_display[:max_display_chars] + "..."
                
                print(f"\n📝 INPUT TEXT (tokens 0-510):")
                print(f"   {input_text_display}")
                
                print(f"\n🎯 TARGET TEXT (tokens 1-511 - what should come next):")
                print(f"   {target_text_display}")
                
                print(f"\n🤖 PREDICTED TEXT (model's predictions for tokens 1-511):")
                print(f"   {predicted_text_display}")
                
                # Calculate accuracy for this sequence
                import numpy as np
                correct_predictions = np.sum(np.array(predicted_display) == np.array(target_display))
                accuracy = correct_predictions / len(predicted_display) * 100
                
                print(f"\n📈 Sequence Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(predicted_display)} tokens)")
                print("="*80 + "\n")
            
            # Accumulate metrics - use stored loss_value from autocast context
            # This ensures we only keep the scalar value, not the tensor
            total_loss += loss_value * batch.size(0)
            total_tokens += batch.numel()
            num_batches += 1
            
            # Explicitly delete GPU tensors to free memory immediately
            # This is especially important since eval batches are transferred from CPU
            del outputs  # Delete the entire outputs dictionary
            del loss     # Delete the loss tensor
            del batch    # Delete the GPU batch tensor
            
            # Clear GPU cache after each batch to ensure memory is freed
            torch.cuda.empty_cache()
        
        # Calculate average metrics
        batch_size = self.config['training']['batch_size']
        avg_loss = total_loss / (num_batches * batch_size)
        perplexity = self.metrics.calculate_perplexity(avg_loss)
        
        metrics = {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
        
        # Clear GPU cache to release any remaining memory
        torch.cuda.empty_cache()
        
        return metrics
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """Save training checkpoint.
        
        Args:
            checkpoint_name: Optional specific name for checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint = {
            'global_step': self.global_step,
            'current_data_subset_index': self.current_data_subset_index,  # Save which subset index we're on
            'completed_subsets': self.completed_subsets,  # Save completed subset indices
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_size': self.model_size or self.config.get('default_model_size', 'medium'),
            'metrics_summary': self.metrics.get_summary()
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as 'latest' for easy resuming
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best model separately
        if self.best_val_loss < float('inf'):
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            if not best_path.exists() or self.metrics.get_latest_metrics().get('val_loss', float('inf')) <= self.best_val_loss:
                torch.save(checkpoint, best_path)
        
        print(f"\n💾 Checkpoint saved: {checkpoint_path.name}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"\nLoading checkpoint: {checkpoint_path}")
        
        # Note: weights_only=False is safe here since we're loading our own trained checkpoints
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Check if checkpoint has config and verify compatibility
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            
            # Get model sizes
            checkpoint_model_size = checkpoint.get('model_size', None)
            current_model_size = self.model_size or self.config.get('default_model_size', 'medium')
            
            # Check if model sizes match
            if checkpoint_model_size and checkpoint_model_size != current_model_size:
                print(f"\n  ⚠️  WARNING: Model size mismatch!")
                print(f"  Checkpoint was trained with '{checkpoint_model_size}' model")
                print(f"  Current model size: '{current_model_size}'")
                response = input("\n  Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("  Checkpoint loading cancelled.")
                    sys.exit(1)
            
            # Check critical model architecture parameters
            # Handle both old single-model and new multi-model config formats
            model_params = ['vocab_size', 'hidden_size', 'num_layers', 'num_heads', 'max_position_embeddings']
            mismatches = []
            
            # Get saved model config (handle both formats)
            if 'models' in saved_config:
                saved_model_size = checkpoint_model_size or saved_config.get('default_model_size', 'medium')
                saved_model_config = saved_config['models'].get(saved_model_size, {})
            else:
                saved_model_config = saved_config.get('model', {})
            
            # Get current model config (handle both formats)
            if 'models' in self.config:
                current_model_config = self.config['models'].get(current_model_size, {})
            else:
                current_model_config = self.config.get('model', {})
            
            for param in model_params:
                saved_val = saved_model_config.get(param)
                current_val = current_model_config.get(param)
                if saved_val != current_val:
                    mismatches.append(f"    {param}: checkpoint={saved_val}, current={current_val}")
            
            if mismatches:
                print("\n  ⚠️  WARNING: Model architecture mismatch detected!")
                print("  The checkpoint was trained with different model parameters:")
                for mismatch in mismatches:
                    print(mismatch)
                print("\n  This will likely cause errors when loading model weights.")
                print("  Consider using the same config.json that was used for training.")
                response = input("\n  Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("  Checkpoint loading cancelled.")
                    sys.exit(1)
        
        # Restore model state
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"\n  ❌ ERROR: Failed to load model weights!")
            print(f"  {str(e)}")
            print("\n  This usually happens when the model architecture doesn't match.")
            print("  Make sure you're using the same config.json that was used for training.")
            sys.exit(1)
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore completed subsets list
        self.completed_subsets = checkpoint.get('completed_subsets', [])
        
        # Clear step times for fresh ETA calculation after resume
        self.recent_step_times.clear()
        
        # Restore data subset state if available
        saved_subset_idx = checkpoint.get('current_data_subset_index', 0)
        if not isinstance(saved_subset_idx, int):
            saved_subset_idx = 0
        
        # Check if current subset is already complete and move to next if needed
        if saved_subset_idx in self.completed_subsets:
            if saved_subset_idx < self.num_data_subsets - 1:
                next_subset_idx = saved_subset_idx + 1
                print(f"  Subset {saved_subset_idx+1} already complete, moving to subset {next_subset_idx+1}")
                saved_subset_idx = next_subset_idx
            else:
                print(f"  All subsets completed! Training is finished.")
                saved_subset_idx = self.num_data_subsets - 1  # Stay on last subset
        
        # Recalculate steps dynamically based on train_quarter flag and remaining data
        from dataloader import calculate_fixed_eval_dataloader_stats
        
        # Get num_epochs from config
        num_epochs = self.config['training'].get('num_epochs', 1)
        
        if self.test_mode:
            # Test mode - train on single subset only
            current_subset_stats = calculate_fixed_eval_dataloader_stats(
                batch_size=self.config['training']['batch_size'],
                test_mode=self.test_mode,
                subset_index=saved_subset_idx,
                num_subsets=self.num_data_subsets
            )
            
            steps_in_current_subset = current_subset_stats['train_batches'] // self.gradient_accumulation_steps
            if current_subset_stats['train_batches'] % self.gradient_accumulation_steps != 0:
                steps_in_current_subset += 1
            
            # For test mode, complete the remaining steps of the current subset
            # accounting for the epochs
            self.max_steps = steps_in_current_subset * num_epochs
            self.total_training_steps = self.max_steps
        else:
            # Train on all subsets sequentially
            # Calculate total steps for ALL subsets (not just remaining)
            total_steps = 0
            
            for subset_idx in range(self.num_data_subsets):
                subset_stats = calculate_fixed_eval_dataloader_stats(
                    batch_size=self.config['training']['batch_size'],
                    test_mode=self.test_mode,
                    subset_index=subset_idx,
                    num_subsets=self.num_data_subsets
                )
                steps_in_subset = subset_stats['train_batches'] // self.gradient_accumulation_steps
                if subset_stats['train_batches'] % self.gradient_accumulation_steps != 0:
                    steps_in_subset += 1
                total_steps += steps_in_subset
            
            # Set max_steps for all subsets with epochs
            self.max_steps = total_steps * num_epochs
            self.total_training_steps = self.max_steps
        
        print(f"  Resumed from step {self.global_step}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Data subset: {saved_subset_idx+1}/{self.num_data_subsets}")
        print(f"  Completed subsets: {self.completed_subsets}")
        
        # Calculate progress information
        remaining_steps = self.max_steps - self.global_step
        progress_percent = (self.global_step / self.max_steps * 100) if self.max_steps > 0 else 0
        
        if self.test_mode:
            print(f"  Mode: Test mode (single subset training)")
            print(f"  Progress: {self.global_step}/{self.max_steps} steps ({progress_percent:.1f}%)")
            print(f"  Remaining: {remaining_steps} steps")
        else:
            print(f"  Mode: All {self.num_data_subsets} subsets training ({num_epochs} epoch{'s' if num_epochs > 1 else ''})")
            print(f"  Progress: {self.global_step}/{self.max_steps} steps ({progress_percent:.1f}%)")
            print(f"  Remaining: {remaining_steps} steps")
        
        # Load the appropriate data subset
        if saved_subset_idx is not None:
            self._create_dataloaders(saved_subset_idx)
        
        # Check if already completed
        if self.global_step >= self.max_steps:
            print(f"  WARNING: Already completed {self.global_step} steps (max_steps: {self.max_steps})")
            print(f"  Training will end immediately unless max_steps is increased.")
    
    def train(self):
        """Main training loop with proper gradient accumulation and data subset switching."""
        print(f"\n{'='*70}")
        print(f"Starting Training")
        print(f"{'='*70}")
        
        # Use the gradient accumulation steps stored in __init__
        grad_accum_steps = self.gradient_accumulation_steps
        effective_batch_size = self.config['training']['batch_size'] * grad_accum_steps
        
        print(f"Gradient accumulation: {grad_accum_steps} steps | Effective batch size: {effective_batch_size}\n")
        
        # Initialize accumulation tracking
        accumulation_counter = 0
        accumulated_loss = 0.0
        accumulated_tokens = 0
        iteration_start_time = None  # Track timing across accumulation
        
        # Training loop - continue until max_steps reached
        current_subset_idx = getattr(self, 'current_data_subset_index', 0)
        
        try:
            while self.global_step < self.max_steps:
                # Cycle through data subsets continuously
                for subset_idx in range(current_subset_idx, self.num_data_subsets):
                    
                    # Check if we've reached max_steps before processing new subset
                    if self.global_step >= self.max_steps:
                        break
                    
                    # Skip already completed subsets
                    if subset_idx in self.completed_subsets:
                        print(f"Skipping already completed subset {subset_idx+1}/{self.num_data_subsets}")
                        continue
                    
                    # Load the appropriate data subset
                    if not hasattr(self, 'current_data_subset_index') or self.current_data_subset_index != subset_idx:
                        if self.train_loader is None:
                            # First time loading data
                            self._create_dataloaders(subset_idx)
                        else:
                            # Switch to new subset
                            self._switch_data_subset(subset_idx)
                    
                    print(f"\nTraining on subset {subset_idx+1}/{self.num_data_subsets} (Step {self.global_step}/{self.max_steps})")
                    print(f"{'─'*50}")
                    
                    # Training on current data subset
                    for batch_idx, batch in enumerate(self.train_loader):
                        # Check if we've reached max_steps
                        if self.global_step >= self.max_steps:
                            print(f"\nReached maximum training steps ({self.max_steps})")
                            break
                        # Start timing at the beginning of each accumulation cycle
                        # This captures data loading time for ALL accumulated batches
                        if accumulation_counter == 0:
                            iteration_start_time = time.time()
                        
                        # Forward and backward pass
                        loss_value = self.train_step(batch, grad_accum_steps)
                        
                        # Accumulate metrics
                        accumulated_loss += loss_value
                        accumulated_tokens += batch.numel()
                        accumulation_counter += 1
                        
                        # Check if we should perform optimizer step
                        should_update = (accumulation_counter % grad_accum_steps == 0) or \
                                       (batch_idx == len(self.train_loader) - 1)  # Last batch of current subset
                        
                        if should_update:
                            # Perform optimizer step
                            grad_norm = self.optimizer_step()
                            
                            # Calculate FULL iteration time including all accumulation steps and data loading
                            iteration_time = time.time() - iteration_start_time
                            
                            # Calculate averaged metrics
                            avg_loss = accumulated_loss / accumulation_counter
                            
                            # Update global step (counts optimizer steps, not forward passes)
                            self.global_step += 1
                            
                            # Calculate metrics
                            # Calculate perplexity before creating metrics dict
                            perplexity = self.metrics.calculate_perplexity(avg_loss)
                            
                            step_metrics = {
                                'loss': avg_loss,
                                'perplexity': perplexity,
                                'learning_rate': self.optimizer.param_groups[0]['lr'],
                                'grad_norm': grad_norm
                            }
                            
                            # Update metrics tracker
                            self.metrics.update(
                                step_metrics,
                                tokens=accumulated_tokens,
                                batch_size=effective_batch_size
                            )
                            
                            # Calculate tokens per second based on REAL total time for all accumulated batches
                            tokens_per_second = accumulated_tokens / iteration_time if iteration_time > 0 else 0
                            
                            # Update step times for ETA calculation
                            self.recent_step_times.append(iteration_time)
                            
                            # Calculate ETA based on running average of recent step times
                            eta_str = ""
                            if len(self.recent_step_times) >= 1:  # Start showing ETA immediately
                                # Calculate average time per step
                                avg_step_time = sum(self.recent_step_times) / len(self.recent_step_times)
                                
                                # Calculate remaining steps for current subset
                                remaining_steps = self.max_steps - self.global_step
                                
                                if remaining_steps > 0:
                                    # Calculate ETA in seconds
                                    eta_seconds = avg_step_time * remaining_steps
                                    
                                    # Format ETA as HH:MM:SS or MM:SS
                                    if eta_seconds < 3600:  # Less than 1 hour
                                        eta_minutes = int(eta_seconds // 60)
                                        eta_secs = int(eta_seconds % 60)
                                        eta_str = f" | ETA: {eta_minutes:02d}:{eta_secs:02d}"
                                    else:  # 1 hour or more
                                        eta_hours = int(eta_seconds // 3600)
                                        eta_minutes = int((eta_seconds % 3600) // 60)
                                        eta_secs = int(eta_seconds % 60)
                                        eta_str = f" | ETA: {eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d}"
                                else:
                                    eta_str = " | ETA: Complete"
                            
                            # Create fraction label for display
                            subset_label = f"{subset_idx+1}/{self.num_data_subsets}"
                            
                            # Get current learning rate
                            current_lr = self.optimizer.param_groups[0]['lr']
                            
                            # Print step metrics with data subset indicator, learning rate, and ETA
                            # Time shown is TOTAL time for all gradient accumulation steps
                            print(f"[{subset_label:5s}] Step {self.global_step:5d}/{self.total_training_steps} | "
                                  f"{iteration_time * 1000:6.0f}ms | "
                                  f"Loss: {avg_loss:7.4f} | "
                                  f"LR: {current_lr:.2e} | "
                                  f"PPL: {perplexity:8.2f} | "
                                  f"Tokens/s: {tokens_per_second:8.0f}{eta_str}")
                            
                            # Logging to tensorboard/files
                            if self.global_step % self.config['logging']['logging_steps'] == 0:
                                # Get smoothed metrics
                                smoothed = self.metrics.get_smoothed_metrics()
                                
                                # Override learning_rate with actual value (not smoothed)
                                smoothed['learning_rate'] = current_lr
                                
                                # Log to dual logger
                                self.logger.log_scalars(smoothed, self.global_step, prefix="train")
                            
                            # Validation (based on optimizer steps, not batch index)
                            # validation_interval is already correctly calculated in __init__
                            if self.global_step > 0 and self.global_step % self.validation_interval == 0:
                                print(f"\n{'='*70}")
                                print(f"Validation at step {self.global_step} (on subset {subset_idx+1}/{self.num_data_subsets})")
                                print(f"{'='*70}")
                                
                                # Run validation on current subset's validation split
                                val_metrics = self.validate()
                                
                                # Log validation metrics
                                self.logger.log_scalars(val_metrics, self.global_step, prefix="validation")
                                
                                # Update best model
                                if val_metrics['val_loss'] < self.best_val_loss:
                                    self.best_val_loss = val_metrics['val_loss']
                                    print(f"✨ New best validation loss: {self.best_val_loss:.4f}")
                                
                                # Print validation results
                                print(f"Val Loss: {val_metrics['val_loss']:.4f} | Val PPL: {val_metrics['val_perplexity']:.2f}")
                                print(f"{'='*70}\n")
                                
                                # Save checkpoint after each validation
                                self.save_checkpoint()
                                
                                # Additional GPU memory cleanup after validation
                                # This ensures any remaining cached memory is fully released
                                torch.cuda.empty_cache()
                            
                            # Regular checkpoint saving
                            if self.global_step % self.config['logging']['save_steps'] == 0:
                                self.save_checkpoint()
                            
                            # Reset accumulation
                            accumulation_counter = 0
                            accumulated_loss = 0.0
                            accumulated_tokens = 0
                    
                    # After processing all batches in this subset, mark it as complete
                    if subset_idx not in self.completed_subsets:
                        self.completed_subsets.append(subset_idx)
                        print(f"\n✓ Completed subset {subset_idx+1}/{self.num_data_subsets}")
                        print(f"Saving checkpoint after completing subset {subset_idx+1}...")
                        self.save_checkpoint(f"checkpoint_subset_{subset_idx+1}_complete.pt")
                        
                        # Check if we've completed all subsets
                        if len(self.completed_subsets) == self.num_data_subsets:
                            print(f"\n🎉 All {self.num_data_subsets} subsets completed! Training finished.")
                            break
                
                # Check if we should continue or stop based on test mode
                if self.global_step >= self.max_steps:
                    if self.test_mode:
                        print(f"\nCompleted current subset. Stopping training (test mode).")
                        break
                    else:
                        # In full training mode, we've completed all subsets
                        print(f"\nCompleted all training steps across all subsets.")
                        break
                
                # Reset to first subset for next epoch if not in test mode
                if not self.test_mode and subset_idx == self.num_data_subsets - 1:
                    # Completed all subsets but haven't reached max_steps
                    # Reset for another pass through the data
                    current_subset_idx = 0
                    self.completed_subsets = []  # Reset completed subsets for next epoch
                    print(f"\nStarting new epoch through all {self.num_data_subsets} subsets...")
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print("Saving checkpoint...")
            self.save_checkpoint("checkpoint_interrupted.pt")
        
        except Exception as e:
            print(f"\n\nTraining failed with error: {e}")
            print("Saving checkpoint...")
            self.save_checkpoint("checkpoint_error.pt")
            raise
        
        finally:
            # Final cleanup
            self.logger.close()
            
            # Print final summary
            summary = self.metrics.get_summary()
            print(f"\n{'='*70}")
            print(f"Training Complete")
            print(f"{'='*70}")
            print(f"\nFinal Metrics:")
            print(f"  • Total steps: {self.global_step:,}/{self.max_steps:,} ({100*self.global_step/self.max_steps:.1f}% complete)")
            print(f"  • Total tokens: {summary['progress']['total_tokens']:,}")
            print(f"  • Training time: {summary['progress']['training_hours']:.2f} hours")
            print(f"  • Best validation loss: {self.best_val_loss:.4f}")
            if 'throughput' in summary:
                print(f"  • Tokens/second: {summary['throughput'].get('tokens_per_second', 0):.0f}")
            # Always report GPU memory since we always use CUDA
            print(f"  • Peak memory: {summary['memory']['peak_memory_gb']:.2f} GB")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train transformer language model")
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--model-size', type=str, choices=['small', 'medium', 'large'], default=None,
                       help='Model size to use (small, medium, or large). Defaults to config setting.')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode - train on 1 chunk (1/64 of data)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(
        config_path=args.config,
        model_size=args.model_size,
        test_mode=args.test,
        resume_from=args.resume,
        experiment_name=args.name
    )
    
    trainer.train()


if __name__ == "__main__":
    main()