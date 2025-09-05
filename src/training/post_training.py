#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) script for instruction following using LIMA dataset."""

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Import project modules
from src.utils.model import create_model, TransformerLM
from src.dataset_preparation.tokenizer import CodeLlamaTokenizer
from src.utils.logging_utils import DualLogger
from src.utils.scheduler import get_cosine_schedule_with_warmup
from src.utils.metrics import MetricsTracker

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class SFTTrainer:
    """Supervised Fine-Tuning trainer for instruction following."""
    
    def __init__(
        self,
        checkpoint_path: str,
        num_epochs: int = 1,
        learning_rate: float = 1e-6,
        batch_size: int = 16,
        experiment_name: Optional[str] = None
    ):
        """Initialize SFT trainer.
        
        Args:
            checkpoint_path: Path to pretrained model checkpoint
            num_epochs: Number of training epochs (default 1)
            learning_rate: Learning rate for fine-tuning (default 5e-5)
            batch_size: Batch size for training (default 8)
            experiment_name: Name for this SFT run
        """
        print(f"\n{'='*80}")
        print(f"Supervised Fine-Tuning (SFT) Pipeline")
        print(f"{'='*80}")
        
        self.checkpoint_path = Path(checkpoint_path)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Verify checkpoint exists
        if not self.checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {self.checkpoint_path}")
            sys.exit(1)
        
        # Setup device - CUDA is required
        if not torch.cuda.is_available():
            print("❌ CUDA is not available. GPU is required for training.")
            sys.exit(1)
        
        self.device = torch.device('cuda')
        print(f"Device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Generate experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"sft_lima_{timestamp}"
        self.experiment_name = experiment_name
        
        # Setup checkpoint directory for SFT
        self.checkpoint_dir = Path("checkpoints") / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"SFT checkpoint directory: {self.checkpoint_dir}")
        
        # Initialize logging
        self.logger = DualLogger(
            log_dir="logs",
            experiment_name=self.experiment_name
        )
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker(window_size=50)
        
        # Load pretrained model
        print(f"\nLoading pretrained model: {self.checkpoint_path}")
        self.model, self.pretrained_config, self.model_size = self._load_pretrained_model()
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = CodeLlamaTokenizer()
        tokenizer_path = self.pretrained_config['paths']['tokenizer_dir']
        self.tokenizer.load(tokenizer_path)
        print(f"Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")
        
        # Load LIMA dataset
        print("\nLoading LIMA dataset...")
        self.train_loader, self.val_loader, self.dataset_info = self._load_lima_data()
        
        # Setup optimizer
        print("\nSetting up optimizer...")
        self.optimizer = self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.recent_step_times = deque(maxlen=10)
        
        # Log configuration
        sft_config = {
            'pretrained_checkpoint': str(self.checkpoint_path),
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'dataset': self.dataset_info,
            'model_size': self.model_size,
            'experiment_name': self.experiment_name
        }
        self.logger.log_config(sft_config)
        
        # Save SFT configuration
        sft_config_path = self.checkpoint_dir / "sft_config.json"
        with open(sft_config_path, 'w') as f:
            json.dump(sft_config, f, indent=2)
        
    def _load_pretrained_model(self) -> Tuple[TransformerLM, dict, str]:
        """Load and verify pretrained model from checkpoint.
        
        Returns:
            Tuple of (model, config, model_size)
        """
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract configuration
        if 'config' not in checkpoint:
            print("❌ No configuration found in checkpoint!")
            sys.exit(1)
        
        config = checkpoint['config']
        model_size = checkpoint.get('model_size', config.get('default_model_size', 'medium'))
        
        # Get model configuration
        if 'models' in config:
            model_config = config['models'].get(model_size)
            if not model_config:
                print(f"❌ Model size '{model_size}' not found in config!")
                sys.exit(1)
        else:
            print("❌ Invalid configuration format in checkpoint!")
            sys.exit(1)
        
        # Create model with same configuration
        print(f"Creating model with size: {model_size}")
        model = create_model("config.json", model_size=model_size)
        
        # Load pretrained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to GPU
        model = model.to(self.device)
        
        # Calculate and display model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model architecture verified: {model_size}")
        print(f"  Total parameters: {total_params/1e6:.1f}M")
        print(f"  Trainable parameters: {trainable_params/1e6:.1f}M")
        print(f"  Layers: {model_config['num_layers']}")
        print(f"  Hidden size: {model_config['hidden_size']}")
        print(f"  Attention heads: {model_config['num_heads']}")
        
        # Log pretrained checkpoint info
        if 'global_step' in checkpoint:
            print(f"  Pretrained for {checkpoint['global_step']:,} steps")
        if 'best_val_loss' in checkpoint:
            print(f"  Pretrained validation loss: {checkpoint['best_val_loss']:.4f}")
        
        return model, config, model_size
    
    def _load_lima_data(self) -> Tuple[DataLoader, DataLoader, dict]:
        """Load LIMA tokenized dataset into memory.
        
        Returns:
            Tuple of (train_loader, val_loader, dataset_info)
        """
        data_dir = Path("/home/andrea/Desktop/data/post-training")
        
        # Load tokenized examples
        tokenized_path = data_dir / "tokenized_examples.npy"
        if not tokenized_path.exists():
            print(f"❌ Tokenized data not found: {tokenized_path}")
            print("Please run lima_tokenizer.py first to prepare the dataset.")
            sys.exit(1)
        
        # Load metadata
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load tokenized sequences
        tokenized_data = np.load(tokenized_path)
        num_examples = len(tokenized_data)
        
        print(f"✓ Loaded LIMA dataset: {num_examples} examples")
        if metadata:
            print(f"  Average tokens: {metadata.get('avg_tokens', 'N/A'):.1f}")
            print(f"  Max length: {metadata.get('max_length', 2048)}")
        
        # Convert to PyTorch tensors and move to GPU
        # Using int32 instead of int64 to save memory (vocab size < 2^31)
        all_data = torch.from_numpy(tokenized_data).int().to(self.device)
        
        # Create train/val split (90/10)
        val_size = max(1, int(num_examples * 0.1))
        train_size = num_examples - val_size
        
        # Use deterministic split
        torch.manual_seed(42)
        indices = torch.randperm(num_examples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_data = all_data[train_indices]
        val_data = all_data[val_indices]
        
        print(f"  Train examples: {train_size}")
        print(f"  Validation examples: {val_size}")
        
        # Create TensorDatasets
        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        # Calculate total training steps
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.num_epochs
        
        dataset_info = {
            'total_examples': num_examples,
            'train_examples': train_size,
            'val_examples': val_size,
            'steps_per_epoch': steps_per_epoch,
            'total_steps': total_steps,
            'metadata': metadata
        }
        
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total training steps: {total_steps}")
        
        # Store for later use
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_steps
        
        return train_loader, val_loader, dataset_info
    
    def _setup_optimizer(self) -> AdamW:
        """Setup AdamW optimizer with weight decay."""
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
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(
            optimizer_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True  # Use fused optimizer for better performance
        )
        
        print(f"Optimizer: AdamW (lr={self.learning_rate:.1e})")
        
        return optimizer
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Perform single training step.
        
        Args:
            batch: Input token IDs
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Batch is already on GPU
        # Unpack from TensorDataset format
        input_ids = batch[0]
        
        # Forward pass with autocast
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        return loss.item(), grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and show sample generation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch[0]
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = self.metrics.calculate_perplexity(avg_loss)
        
        # Sample generation for quality check (use first validation example)
        if len(self.val_loader) > 0:
            first_batch = next(iter(self.val_loader))
            input_ids = first_batch[0][0]  # First sequence
            
            # Find a good prompt position (first 50 tokens)
            prompt_length = min(50, len(input_ids))
            prompt = input_ids[:prompt_length].unsqueeze(0)
            
            # Generate continuation
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                generated = self.model.generate(
                    prompt,
                    max_length=150,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
            
            # Decode and display
            prompt_text = self.tokenizer.decode(prompt[0].cpu().tolist(), skip_special_tokens=False)
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=False)
            
            print("\n" + "="*70)
            print("📝 Sample Generation")
            print("="*70)
            print(f"Prompt: {prompt_text[:100]}...")
            print(f"Generated: {generated_text[len(prompt_text):200]}...")
            print("="*70)
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """Save SFT checkpoint.
        
        Args:
            checkpoint_name: Optional specific name for checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{self.current_epoch}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data - matching pre-training format for compatibility
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': None,  # SFT doesn't use scheduler, but included for format consistency
            'best_val_loss': self.best_val_loss,
            'config': self.pretrained_config,  # Full config for compatibility with Inference.py and benchmark.py
            'model_size': self.model_size,  # Root level for compatibility
            'pretrained_checkpoint': str(self.checkpoint_path),
            'sft_config': {
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'model_size': self.model_size  # Keep in sft_config too for reference
            },
            'metrics_summary': self.metrics.get_summary()
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as 'latest'
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if self.best_val_loss < float('inf'):
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            if not best_path.exists() or checkpoint.get('val_loss', float('inf')) <= self.best_val_loss:
                torch.save(checkpoint, best_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path.name}")
    
    def train(self):
        """Main training loop for SFT."""
        print(f"\n{'='*80}")
        print(f"Starting Supervised Fine-Tuning")
        print(f"Epochs: {self.num_epochs} | Batch size: {self.batch_size} | Learning rate: {self.learning_rate:.1e}")
        print(f"{'='*80}\n")
        
        try:
            for epoch in range(1, self.num_epochs + 1):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                print(f"Epoch {epoch}/{self.num_epochs}")
                print("-" * 40)
                
                # Training loop
                epoch_loss = 0
                for batch_idx, batch in enumerate(self.train_loader, 1):
                    step_start_time = time.time()
                    
                    # Training step
                    loss, grad_norm = self.train_step(batch)
                    
                    # Update metrics
                    self.global_step += 1
                    epoch_loss += loss
                    
                    # Calculate step time
                    step_time = time.time() - step_start_time
                    self.recent_step_times.append(step_time)
                    
                    # Calculate metrics
                    current_lr = self.learning_rate  # Fixed learning rate
                    perplexity = self.metrics.calculate_perplexity(loss)
                    tokens_per_sec = batch[0].numel() / step_time
                    
                    # Update metrics tracker
                    step_metrics = {
                        'loss': loss,
                        'perplexity': perplexity,
                        'learning_rate': current_lr,
                        'grad_norm': grad_norm
                    }
                    self.metrics.update(step_metrics, tokens=batch[0].numel(), batch_size=self.batch_size)
                    
                    # Calculate ETA
                    if len(self.recent_step_times) > 0:
                        avg_step_time = sum(self.recent_step_times) / len(self.recent_step_times)
                        remaining_steps = (self.num_epochs - epoch + 1) * self.steps_per_epoch - batch_idx
                        eta_seconds = avg_step_time * remaining_steps
                        
                        if eta_seconds < 3600:
                            eta_str = f"{int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
                        else:
                            eta_hours = int(eta_seconds // 3600)
                            eta_minutes = int((eta_seconds % 3600) // 60)
                            eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{int(eta_seconds%60):02d}"
                    else:
                        eta_str = "N/A"
                    
                    # Print progress every step (since dataset is small)
                    print(f"Step {batch_idx:3d}/{self.steps_per_epoch} | "
                          f"{step_time*1000:6.0f}ms | "
                          f"Loss: {loss:7.4f} | "
                          f"LR: {current_lr:.1e} | "
                          f"PPL: {perplexity:8.2f} | "
                          f"Tokens/s: {tokens_per_sec:8.0f} | "
                          f"ETA: {eta_str}")
                    
                    # Log to tensorboard
                    if self.global_step % 5 == 0:
                        smoothed = self.metrics.get_smoothed_metrics()
                        self.logger.log_scalars(smoothed, self.global_step, prefix="train")
                    
                    # Validation at end of epoch
                    if batch_idx == self.steps_per_epoch:
                        print(f"\n{'='*70}")
                        print(f"Validation at step {self.global_step}")
                        print(f"{'='*70}")
                        
                        val_metrics = self.validate()
                        
                        # Log validation metrics
                        self.logger.log_scalars(val_metrics, self.global_step, prefix="validation")
                        
                        # Update best model
                        if val_metrics['val_loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['val_loss']
                            print(f"✨ New best validation loss: {self.best_val_loss:.4f}")
                            self.save_checkpoint("checkpoint_best.pt")
                        
                        print(f"Val Loss: {val_metrics['val_loss']:.4f} | Val PPL: {val_metrics['val_perplexity']:.2f}")
                        print(f"{'='*70}\n")
                
                # End of epoch
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / self.steps_per_epoch
                
                print(f"\nEpoch {epoch} complete!")
                print(f"  Time: {epoch_time:.1f}s")
                print(f"  Average loss: {avg_epoch_loss:.4f}")
                print(f"  Best val loss: {self.best_val_loss:.4f}")
                
                # Save epoch checkpoint
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
                
                print("-" * 40 + "\n")
        
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
            print(f"\n{'='*80}")
            print(f"Training Complete!")
            print(f"{'='*80}")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Final checkpoint: {self.checkpoint_dir / 'checkpoint_latest.pt'}")
            print(f"Best checkpoint: {self.checkpoint_dir / 'checkpoint_best.pt'}")
            print(f"{'='*80}\n")


def main():
    """Main entry point for SFT training."""
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for instruction following")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-6,
                       help="Learning rate for fine-tuning (default: 1e-6)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training (default: 16)")
    parser.add_argument("--name", type=str,
                       help="Experiment name for this SFT run")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SFTTrainer(
        checkpoint_path=args.checkpoint,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        experiment_name=args.name
    )
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSFT interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)