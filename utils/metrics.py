"""Metrics tracking and calculation utilities for training."""

import time
from collections import deque
from typing import Dict, Optional, Deque
import numpy as np
import torch


class MetricsTracker:
    """Track and smooth training metrics over time."""
    
    def __init__(
        self,
        window_size: int = 100,
        warmup_steps: int = 10
    ):
        """Initialize metrics tracker.
        
        Args:
            window_size: Size of smoothing window for metrics
            warmup_steps: Number of steps before showing smoothed metrics
        """
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        
        # Metric storage with deques for efficient windowing
        self.metrics: Dict[str, Deque[float]] = {}
        
        # Timing information
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.total_tokens = 0
        self.total_steps = 0
        
        # Memory tracking
        self.peak_memory_gb = 0.0
        
        # Best metrics tracking
        self.best_metrics: Dict[str, float] = {}
    
    def update(
        self,
        metrics: Dict[str, float],
        tokens: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        """Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric names and values
            tokens: Number of tokens processed in this step
            batch_size: Batch size for this step
        """
        current_time = time.time()
        
        # Update metric histories
        for name, value in metrics.items():
            # Only track numeric values
            if not isinstance(value, (int, float, np.number)):
                print(f"Warning: Skipping non-numeric metric '{name}': {value}")
                continue
            
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.window_size)
            self.metrics[name].append(float(value))  # Ensure it's a float
            
            # Track best metrics (for loss, lower is better)
            if 'loss' in name:
                if name not in self.best_metrics or float(value) < self.best_metrics[name]:
                    self.best_metrics[name] = float(value)
            else:
                if name not in self.best_metrics or float(value) > self.best_metrics[name]:
                    self.best_metrics[name] = float(value)
        
        # Update token and step counts
        if tokens is not None:
            self.total_tokens += tokens
        self.total_steps += 1
        
        # Update timing
        self.last_update_time = current_time
        
        # Update memory usage if on CUDA
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
            self.peak_memory_gb = max(self.peak_memory_gb, current_memory)
    
    def get_smoothed_metrics(self) -> Dict[str, float]:
        """Get smoothed metrics using moving average.
        
        Returns:
            Dictionary of smoothed metric values
        """
        smoothed = {}
        
        for name, values in self.metrics.items():
            if len(values) > 0:
                # Use full history for warmup, then window
                if self.total_steps < self.warmup_steps:
                    smoothed[name] = np.mean(list(values))
                else:
                    smoothed[name] = np.mean(list(values))
        
        return smoothed
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the most recent metric values.
        
        Returns:
            Dictionary of latest metric values
        """
        latest = {}
        for name, values in self.metrics.items():
            if len(values) > 0:
                latest[name] = values[-1]
        return latest
    
    def calculate_throughput(self, time_delta: Optional[float] = None) -> Dict[str, float]:
        """Calculate training throughput metrics.
        
        Args:
            time_delta: Time delta for rate calculation (uses last update if None)
            
        Returns:
            Dictionary with throughput metrics
        """
        if time_delta is None:
            time_delta = time.time() - self.last_update_time
        
        throughput = {}
        
        # Tokens per second
        if self.total_tokens > 0:
            throughput['tokens_per_second'] = self.total_tokens / (time.time() - self.start_time)
        
        # Steps per second
        if self.total_steps > 0:
            throughput['steps_per_second'] = self.total_steps / (time.time() - self.start_time)
        
        # Training time
        throughput['hours_trained'] = (time.time() - self.start_time) / 3600
        
        return throughput
    
    def calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from cross-entropy loss.
        
        Args:
            loss: Cross-entropy loss value
            
        Returns:
            Perplexity value
        """
        try:
            return float(np.exp(loss))
        except OverflowError:
            return float('inf')
    
    def get_summary(self) -> Dict[str, any]:
        """Get comprehensive metrics summary.
        
        Returns:
            Dictionary with all tracked metrics and statistics
        """
        summary = {
            'smoothed': self.get_smoothed_metrics(),
            'latest': self.get_latest_metrics(),
            'best': self.best_metrics.copy(),
            'throughput': self.calculate_throughput(),
            'memory': {
                'peak_memory_gb': self.peak_memory_gb,
                'current_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            },
            'progress': {
                'total_steps': self.total_steps,
                'total_tokens': self.total_tokens,
                'training_hours': (time.time() - self.start_time) / 3600
            }
        }
        
        # Add perplexity if loss is available
        if 'loss' in summary['smoothed']:
            summary['smoothed']['perplexity'] = self.calculate_perplexity(summary['smoothed']['loss'])
        if 'loss' in summary['latest']:
            summary['latest']['perplexity'] = self.calculate_perplexity(summary['latest']['loss'])
        
        return summary
    
    def format_metrics(self, metrics: Optional[Dict[str, float]] = None) -> str:
        """Format metrics for console display.
        
        Args:
            metrics: Metrics to format (uses smoothed if None)
            
        Returns:
            Formatted string for display
        """
        if metrics is None:
            metrics = self.get_smoothed_metrics()
        
        # Build formatted string
        parts = []
        
        # Loss and perplexity
        if 'loss' in metrics:
            parts.append(f"Loss: {metrics['loss']:.4f}")
            perplexity = self.calculate_perplexity(metrics['loss'])
            if perplexity != float('inf'):
                parts.append(f"PPL: {perplexity:.2f}")
        
        # Validation metrics
        if 'val_loss' in metrics:
            parts.append(f"Val Loss: {metrics['val_loss']:.4f}")
            val_perplexity = self.calculate_perplexity(metrics['val_loss'])
            if val_perplexity != float('inf'):
                parts.append(f"Val PPL: {val_perplexity:.2f}")
        
        # Learning rate
        if 'learning_rate' in metrics:
            parts.append(f"LR: {metrics['learning_rate']:.2e}")
        
        # Gradient norm
        if 'grad_norm' in metrics:
            parts.append(f"Grad: {metrics['grad_norm']:.3f}")
        
        return " | ".join(parts)
    
    def reset(self):
        """Reset all metrics tracking."""
        self.metrics.clear()
        self.best_metrics.clear()
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.total_tokens = 0
        self.total_steps = 0
        self.peak_memory_gb = 0.0


class EarlyStopping:
    """Early stopping handler for training."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min'
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of steps to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better) or 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_step = 0
        self.should_stop = False
    
    def update(self, value: float, step: int) -> bool:
        """Update early stopping state.
        
        Args:
            value: Current metric value
            step: Current step
            
        Returns:
            True if training should stop
        """
        improved = False
        
        if self.mode == 'min':
            if value < self.best_value - self.min_delta:
                improved = True
        else:
            if value > self.best_value + self.min_delta:
                improved = True
        
        if improved:
            self.best_value = value
            self.best_step = step
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.best_step = 0
        self.should_stop = False