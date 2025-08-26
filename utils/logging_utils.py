"""Dual logging system for training metrics (TensorBoard + raw JSON)."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class DualLogger:
    """Logger that writes to both TensorBoard and raw JSON files for matplotlib."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        flush_interval: int = 100
    ):
        """Initialize dual logging system.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name for this training run
            flush_interval: How often to flush metrics to disk
        """
        self.log_dir = Path(log_dir)
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"run_{timestamp}"
        self.experiment_name = experiment_name
        
        # Create directories
        self.tensorboard_dir = self.log_dir / "tensorboard" / experiment_name
        self.raw_metrics_dir = self.log_dir / "raw_metrics" / experiment_name
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.raw_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # Initialize raw metrics storage
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.flush_interval = flush_interval
        self.metrics_count = 0
        
        # Paths for different metric files
        self.scalar_metrics_file = self.raw_metrics_dir / "scalar_metrics.jsonl"
        self.histogram_data_file = self.raw_metrics_dir / "histogram_data.jsonl"
        self.config_file = self.raw_metrics_dir / "config.json"
        print(f"To view TensorBoard: tensorboard --logdir {self.tensorboard_dir.parent}")
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar value to both TensorBoard and raw JSON.
        
        Args:
            name: Metric name (e.g., 'loss/train')
            value: Scalar value
            step: Training step
        """
        # Write to TensorBoard
        self.writer.add_scalar(name, value, step)
        
        # Add to raw metrics buffer
        metric_entry = {
            'timestamp': time.time(),
            'step': step,
            'name': name,
            'value': float(value)
        }
        self.metrics_buffer.append(metric_entry)
        
        # Flush if needed
        self.metrics_count += 1
        if self.metrics_count % self.flush_interval == 0:
            self.flush_scalar_metrics()
    
    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log multiple scalar values at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name
            self.log_scalar(full_name, value, step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log histogram data for parameter distributions.
        
        Args:
            name: Histogram name (e.g., 'weights/layer1')
            values: Tensor of values
            step: Training step
        """
        # Write to TensorBoard
        self.writer.add_histogram(name, values, step)
        
        # Save raw data for matplotlib (sample if too large)
        values_np = values.detach().cpu().numpy().flatten()
        if len(values_np) > 10000:
            # Sample for storage efficiency
            indices = np.random.choice(len(values_np), 10000, replace=False)
            values_np = values_np[indices]
        
        histogram_entry = {
            'timestamp': time.time(),
            'step': step,
            'name': name,
            'values': values_np.tolist(),
            'stats': {
                'mean': float(values_np.mean()),
                'std': float(values_np.std()),
                'min': float(values_np.min()),
                'max': float(values_np.max()),
                'count': len(values_np)
            }
        }
        
        # Write histogram data immediately (less frequent than scalars)
        with open(self.histogram_data_file, 'a') as f:
            f.write(json.dumps(histogram_entry) + '\n')
    
    def log_model_gradients(self, model: torch.nn.Module, step: int):
        """Log gradient statistics for model parameters.
        
        Args:
            model: PyTorch model
            step: Training step
        """
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
                
                # Log individual parameter gradients (sparingly)
                if step % 100 == 0:
                    self.log_histogram(f"gradients/{name}", param.grad.data, step)
        
        if grad_norms:
            # Log aggregate gradient statistics
            self.log_scalar("gradients/mean_norm", np.mean(grad_norms), step)
            self.log_scalar("gradients/max_norm", np.max(grad_norms), step)
            self.log_scalar("gradients/min_norm", np.min(grad_norms), step)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Log current learning rate from optimizer.
        
        Args:
            optimizer: PyTorch optimizer
            step: Training step
        """
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            group_name = f"group_{i}" if len(optimizer.param_groups) > 1 else "main"
            self.log_scalar(f"learning_rate/{group_name}", lr, step)
    
    def log_text(self, name: str, text: str, step: int):
        """Log text output (e.g., generated samples).
        
        Args:
            name: Text identifier
            text: Text content
            step: Training step
        """
        # Write to TensorBoard
        self.writer.add_text(name, text, step)
        
        # Save to separate text file
        text_file = self.raw_metrics_dir / f"{name.replace('/', '_')}.txt"
        with open(text_file, 'a') as f:
            f.write(f"[Step {step}] {text}\n\n")
    
    def log_config(self, config: Dict[str, Any]):
        """Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
        """
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Also log to TensorBoard as text
        config_str = json.dumps(config, indent=2)
        self.writer.add_text("config", f"```json\n{config_str}\n```", 0)
    
    def flush_scalar_metrics(self):
        """Flush buffered scalar metrics to disk."""
        if self.metrics_buffer:
            with open(self.scalar_metrics_file, 'a') as f:
                for entry in self.metrics_buffer:
                    f.write(json.dumps(entry) + '\n')
            self.metrics_buffer = []
    
    def close(self):
        """Close logging resources and flush remaining data."""
        # Flush any remaining metrics
        self.flush_scalar_metrics()
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\nLogging complete. Metrics saved to:")
        print(f"  • TensorBoard: {self.tensorboard_dir}")
        print(f"  • Raw metrics: {self.raw_metrics_dir}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Load and summarize logged metrics for analysis.
        
        Returns:
            Dictionary with metric summaries
        """
        metrics_summary = {
            'scalar_metrics': [],
            'histogram_stats': [],
            'config': None
        }
        
        # Load scalar metrics
        if self.scalar_metrics_file.exists():
            with open(self.scalar_metrics_file, 'r') as f:
                for line in f:
                    metrics_summary['scalar_metrics'].append(json.loads(line))
        
        # Load histogram statistics
        if self.histogram_data_file.exists():
            with open(self.histogram_data_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Store only statistics, not full values
                    metrics_summary['histogram_stats'].append({
                        'timestamp': entry['timestamp'],
                        'step': entry['step'],
                        'name': entry['name'],
                        'stats': entry['stats']
                    })
        
        # Load configuration
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                metrics_summary['config'] = json.load(f)
        
        return metrics_summary