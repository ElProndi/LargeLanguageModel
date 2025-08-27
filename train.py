#!/usr/bin/env python3
"""Main training script for transformer language model."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Import project modules
from model import create_model
from dataloader import create_simple_train_val_dataloaders, destroy_dataloaders, get_memory_usage, calculate_dataloader_stats
from utils import DualLogger, get_cosine_schedule_with_warmup, MetricsTracker

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


class Trainer:
    """Main trainer class for language model training."""
    
    def __init__(
        self,
        config_path: str = "config.json",
        test_mode: bool = False,
        resume_from: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration file
            test_mode: Use smaller dataset for testing
            resume_from: Path to checkpoint to resume from
            experiment_name: Name for this training run
        """
        print(f"Initializing Transformer Language Model Trainer")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.test_mode = test_mode
        if test_mode:
            print("🧪 Running in TEST MODE - using reduced dataset and steps")
            self.config['training']['max_steps'] = 100  # Only 100 steps for test mode
            self.config['logging']['eval_steps'] = 50  # Not used anymore, validation_interval is used instead
            self.config['logging']['save_steps'] = 50   # Save checkpoint every 50 steps in test mode
            self.config['logging']['logging_steps'] = 5
        
        # Setup device - CUDA is required
        if not torch.cuda.is_available():
            print("❌ CUDA is not available. GPU is required for training.")
            print("Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed.")
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
        
        # Initialize dataloader variables (will be created dynamically)
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.current_data_subset = None
        self.dataloader_info = None
        
        # Initialize training state early for calculations
        self.global_step = 0
        self.max_steps = self.config['training']['max_steps']
        self.best_val_loss = float('inf')
        
        # Get gradient accumulation steps early for correct calculations
        self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        # Calculate training steps without loading data (optimized)
        # This avoids loading ~13GB into GPU memory just to count batches
        print("Calculating training steps...", end='')
        
        # Get statistics for first fourth
        first_fourth_stats = calculate_dataloader_stats(
            batch_size=self.config['training']['batch_size'],
            val_split=0.1,
            test_mode=self.test_mode,
            file_subset='first_fourth'
        )
        
        # Get statistics for second fourth
        second_fourth_stats = calculate_dataloader_stats(
            batch_size=self.config['training']['batch_size'],
            val_split=0.1,
            test_mode=self.test_mode,
            file_subset='second_fourth'
        )
        
        # Get statistics for third fourth
        third_fourth_stats = calculate_dataloader_stats(
            batch_size=self.config['training']['batch_size'],
            val_split=0.1,
            test_mode=self.test_mode,
            file_subset='third_fourth'
        )
        
        # Get statistics for fourth fourth
        fourth_fourth_stats = calculate_dataloader_stats(
            batch_size=self.config['training']['batch_size'],
            val_split=0.1,
            test_mode=self.test_mode,
            file_subset='fourth_fourth'
        )
        
        # Calculate batch counts - we have 4 fourths, sum their batches
        # Note: The fourths may have slightly different batch counts due to file distribution
        self.batches_per_epoch = (first_fourth_stats['train_batches'] + 
                                   second_fourth_stats['train_batches'] + 
                                   third_fourth_stats['train_batches'] + 
                                   fourth_fourth_stats['train_batches'])
        
        # With gradient accumulation, optimizer steps = batches / accumulation_steps
        self.steps_per_epoch = self.batches_per_epoch // self.gradient_accumulation_steps
        # Handle remainder batches (last incomplete accumulation in epoch)
        if self.batches_per_epoch % self.gradient_accumulation_steps != 0:
            self.steps_per_epoch += 1
        
        # Set total training steps to max_steps from config
        self.total_training_steps = self.max_steps
        
        # Validation frequency: every 1000 steps (or less for test mode)
        if self.test_mode:
            self.validation_interval = 50  # Validate every 50 steps in test mode
        else:
            self.validation_interval = max(1000, self.max_steps // 10)  # At least every 1000 steps
        
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
        # Create model - automatically moved to GPU in model.py
        model = create_model(self.config.get('config_path', 'config.json'))
        # No need to call model.to(device) - model is already on GPU
        
        # Model compilation (commented out by default)
        # Uncomment to enable torch.compile for faster training:

        #print("Compiling model with torch.compile()...")
        #model = torch.compile(model)
        
        return model
    
    def _create_dataloaders(self, file_subset: str = 'first_fourth') -> Tuple[DataLoader, DataLoader, Dict]:
        """Create train and validation data loaders for specified subset.
        
        Args:
            file_subset: Which files to load ('first_fourth', 'second_fourth', 'third_fourth', or 'fourth_fourth')
        
        Returns:
            Tuple of (train_loader, val_loader, info_dict)
        """
        print(f"\nLoading data subset: {file_subset}")
        
        # Create dataloaders for the specified subset
        train_loader, val_loader, info = create_simple_train_val_dataloaders(
            batch_size=self.config['training']['batch_size'],
            val_split=0.1,  # 10% validation split within subset
            shuffle_train=True,
            test_mode=self.test_mode,
            verbose=True,
            seed=42,  # Consistent split across runs
            file_subset=file_subset  # Load only specified fourth
        )
        
        # Store references
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = info['train_dataset']
        self.val_dataset = info['val_dataset']
        self.current_data_subset = file_subset
        self.dataloader_info = info
        
        # Show summary after loading
        mem_stats_after = get_memory_usage()
        if 'gpu_allocated_gb' in mem_stats_after:
            gpu_mem = mem_stats_after['gpu_allocated_gb']
            print(f"  Loaded {info['num_files_loaded']} files | "
                  f"Train: {info['train_size']:,} on {info['train_device']} | "
                  f"Val: {info['val_size']:,} on {info['val_device']} | "
                  f"GPU: {gpu_mem:.1f}GB")
        
        return train_loader, val_loader, info
    
    def _switch_data_subset(self, new_subset: str):
        """Switch to a different data subset, destroying old dataloaders first.
        
        Args:
            new_subset: Which subset to switch to ('first_fourth', 'second_fourth', 'third_fourth', or 'fourth_fourth')
        """
        if self.current_data_subset == new_subset:
            print(f"Already using {new_subset}, skipping switch")
            return
        
        print(f"\nSwitching data subset: {self.current_data_subset} → {new_subset}")
        
        # Destroy current dataloaders if they exist
        if self.train_loader is not None:
            destroy_dataloaders(
                self.train_loader, 
                self.val_loader, 
                self.dataloader_info,
                verbose=True
            )
            
            # Clear references
            self.train_loader = None
            self.val_loader = None
            self.train_dataset = None
            self.val_dataset = None
            self.dataloader_info = None
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        # Create new dataloaders for the new subset
        self._create_dataloaders(new_subset)
    
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
            eps=1e-8
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
        """Run validation loop with optimized CPU→GPU transfers.
        
        Data is transferred from CPU to GPU with non_blocking=True,
        allowing overlap with computation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        num_batches = 0
        
        # Use subset of validation data for speed
        max_val_batches = 50 if not self.test_mode else 10
        
        for i, batch in enumerate(self.val_loader):
            if i >= max_val_batches:
                break
            
            # Transfer batch from CPU to GPU with non-blocking transfer
            # This allows the transfer to overlap with previous batch computation
            batch = batch.to(self.device, non_blocking=True)
            
            # Ensure transfer is complete before computation
            # This synchronization is necessary when using non_blocking transfers
            if i == 0:
                # Only sync on first batch to establish the pattern
                torch.cuda.synchronize()
            
            # Forward pass
            outputs = self.model(input_ids=batch, labels=batch)
            loss = outputs['loss']
            
            # Accumulate metrics
            total_loss += loss.item() * batch.size(0)
            total_tokens += batch.numel()
            num_batches += 1
        
        # Calculate average metrics
        avg_loss = total_loss / (num_batches * batch.size(0))
        perplexity = self.metrics.calculate_perplexity(avg_loss)
        
        metrics = {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
        
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
            'current_data_subset': self.current_data_subset,  # Save which subset we're on
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
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
        
        # Clean up old checkpoints (keep only last 5)
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last: int = 10):
        """Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent checkpoints to keep (default: 10)
        """
        # Find all step checkpoints
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove old checkpoints
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
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
            
            # Check critical model architecture parameters
            model_params = ['vocab_size', 'hidden_size', 'num_layers', 'num_heads', 'max_position_embeddings']
            mismatches = []
            
            for param in model_params:
                saved_val = saved_config['model'].get(param)
                current_val = self.config['model'].get(param)
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
        
        # Restore data subset state if available
        saved_subset = checkpoint.get('current_data_subset', 'first_fourth')
        
        print(f"  Resumed from step {self.global_step}/{self.max_steps}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Data subset: {saved_subset}")
        
        # Load the appropriate data subset
        if saved_subset:
            self._create_dataloaders(saved_subset)
        
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
        data_fourths = ['first_fourth', 'second_fourth', 'third_fourth', 'fourth_fourth']
        current_fourth_idx = data_fourths.index(self.current_data_subset) if hasattr(self, 'current_data_subset') and self.current_data_subset else 0
        
        try:
            while self.global_step < self.max_steps:
                # Cycle through data fourths continuously
                for fourth_idx in range(current_fourth_idx, len(data_fourths)):
                    data_fourth = data_fourths[fourth_idx]
                    
                    # Check if we've reached max_steps before processing new fourth
                    if self.global_step >= self.max_steps:
                        break
                    
                    # Load the appropriate data subset
                    if self.current_data_subset != data_fourth:
                        if self.train_loader is None:
                            # First time loading data
                            self._create_dataloaders(data_fourth)
                        else:
                            # Switch to new subset
                            self._switch_data_subset(data_fourth)
                    
                    print(f"\nTraining on {data_fourth} (Step {self.global_step}/{self.max_steps})")
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
                            
                            # Print step metrics with data subset indicator
                            # Time shown is TOTAL time for all gradient accumulation steps
                            print(f"[{data_fourth:12s}] Step {self.global_step:5d}/{self.total_training_steps} | "
                                  f"{iteration_time * 1000:6.0f}ms | "
                                  f"Loss: {avg_loss:7.4f} | "
                                  f"PPL: {perplexity:8.2f} | "
                                  f"Tokens/s: {tokens_per_second:8.0f}")
                            
                            # Logging to tensorboard/files
                            if self.global_step % self.config['logging']['logging_steps'] == 0:
                                # Get smoothed metrics
                                smoothed = self.metrics.get_smoothed_metrics()
                                
                                # Log to dual logger
                                self.logger.log_scalars(smoothed, self.global_step, prefix="train")
                            
                            # Validation (based on optimizer steps, not batch index)
                            # validation_interval is already correctly calculated in __init__
                            if self.global_step > 0 and self.global_step % self.validation_interval == 0:
                                print(f"\n{'='*70}")
                                print(f"Validation at step {self.global_step} (on {data_fourth})")
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
                            
                            # Regular checkpoint saving
                            if self.global_step % self.config['logging']['save_steps'] == 0:
                                self.save_checkpoint()
                            
                            # Reset accumulation
                            accumulation_counter = 0
                            accumulated_loss = 0.0
                            accumulated_tokens = 0
                
                # Reset to first_fourth after completing all fourths
                current_fourth_idx = 0
        
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
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with smaller dataset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(
        config_path=args.config,
        test_mode=args.test,
        resume_from=args.resume,
        experiment_name=args.name
    )
    
    trainer.train()


if __name__ == "__main__":
    main()