"""Learning rate scheduling utilities for transformer training."""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.01,
    cosine_target_ratio: float = 1.0,
    last_epoch: int = -1
) -> LambdaLR:
    """Create a schedule with a learning rate that decreases following the values of the cosine function.
    
    The schedule has three phases:
    1. Linear warmup from 0 to peak learning rate
    2. Cosine annealing from peak to minimum (until cosine_target_ratio of training)
    3. Constant minimum learning rate (after cosine_target_ratio)
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (0.5 = half cosine)
        min_lr_ratio: Minimum learning rate as ratio of initial (e.g., 0.01 = 1% of initial)
        cosine_target_ratio: Fraction of training at which to reach minimum LR (e.g., 0.8 = 80%)
        last_epoch: The index of last epoch when resuming training
        
    Returns:
        LambdaLR scheduler with the appropriate schedule
    """
    
    # Calculate the step at which cosine reaches minimum
    cosine_end_step = int(num_training_steps * cosine_target_ratio)
    
    def lr_lambda(current_step: int) -> float:
        """Calculate learning rate multiplier for current step.
        
        Args:
            current_step: Current training step
            
        Returns:
            Learning rate multiplier (0 to 1)
        """
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Check if we're past the cosine annealing phase
        if current_step >= cosine_end_step:
            # Maintain minimum learning rate
            return min_lr_ratio
        
        # Cosine annealing phase (adjusted to reach minimum at cosine_end_step)
        progress = float(current_step - num_warmup_steps) / float(max(1, cosine_end_step - num_warmup_steps))
        
        # Calculate cosine multiplier
        cosine_mult = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # Apply minimum learning rate
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_mult
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
) -> LambdaLR:
    """Create a schedule with a learning rate that decreases linearly after warmup.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial
        last_epoch: The index of last epoch when resuming training
        
    Returns:
        LambdaLR scheduler with linear decay
    """
    
    def lr_lambda(current_step: int) -> float:
        """Calculate learning rate multiplier for current step."""
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 1.0 - progress)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """Create a schedule with a constant learning rate after warmup.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: Number of steps for the warmup phase
        last_epoch: The index of last epoch when resuming training
        
    Returns:
        LambdaLR scheduler with constant learning rate after warmup
    """
    
    def lr_lambda(current_step: int) -> float:
        """Calculate learning rate multiplier for current step."""
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_inverse_sqrt_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """Create a schedule with an inverse square root decay after warmup.
    
    This schedule is commonly used in transformer training (e.g., in the original Transformer paper).
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: Number of steps for the warmup phase
        last_epoch: The index of last epoch when resuming training
        
    Returns:
        LambdaLR scheduler with inverse sqrt decay
    """
    
    def lr_lambda(current_step: int) -> float:
        """Calculate learning rate multiplier for current step."""
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Inverse square root decay
        return float(num_warmup_steps) ** 0.5 / float(max(1, current_step)) ** 0.5
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WarmupCosineRestartScheduler:
    """Cosine annealing with warm restarts and linear warmup.
    
    This scheduler implements SGDR (Stochastic Gradient Descent with Warm Restarts)
    with an additional linear warmup phase at the beginning.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_period: int,
        period_mult: float = 1.0,
        min_lr_ratio: float = 0.01,
        restart_mult: float = 1.0
    ):
        """Initialize the scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            warmup_steps: Number of warmup steps
            base_period: Base period length for cosine annealing
            period_mult: Multiplier for period length after each restart
            min_lr_ratio: Minimum learning rate as ratio of initial
            restart_mult: Multiplier for learning rate after each restart
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_period = base_period
        self.period_mult = period_mult
        self.min_lr_ratio = min_lr_ratio
        self.restart_mult = restart_mult
        
        self.current_step = 0
        self.current_period = base_period
        self.period_start = warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_base_lrs = self.base_lrs.copy()
    
    def step(self):
        """Update the learning rate."""
        self.current_step += 1
        
        # Calculate new learning rates
        for param_group, base_lr in zip(self.optimizer.param_groups, self.current_base_lrs):
            if self.current_step < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                # Check for restart
                if self.current_step >= self.period_start + self.current_period:
                    self.period_start += self.current_period
                    self.current_period = int(self.current_period * self.period_mult)
                    self.current_base_lrs = [lr * self.restart_mult for lr in self.current_base_lrs]
                
                # Cosine annealing within current period
                progress = (self.current_step - self.period_start) / self.current_period
                cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_mult)
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get the last computed learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return the state of the scheduler as a dict."""
        return {
            'current_step': self.current_step,
            'current_period': self.current_period,
            'period_start': self.period_start,
            'current_base_lrs': self.current_base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """Load the scheduler's state."""
        self.current_step = state_dict['current_step']
        self.current_period = state_dict['current_period']
        self.period_start = state_dict['period_start']
        self.current_base_lrs = state_dict['current_base_lrs']