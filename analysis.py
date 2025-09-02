#!/usr/bin/env python3
"""Training log analysis tool for examining model training dynamics and performance."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import numpy as np
from dataclasses import dataclass


@dataclass
class TrainingPhase:
    """Represents a detected training phase."""
    name: str
    start_step: int
    end_step: int
    avg_loss: float
    loss_reduction_rate: float
    description: str


@dataclass 
class PlateauInfo:
    """Information about a detected training plateau."""
    start_step: int
    end_step: int
    duration: int
    avg_loss: float
    improvement_rate: float
    severity: str  # 'mild', 'moderate', 'severe'


class MetricsProcessor:
    """Process and analyze training metrics."""
    
    @staticmethod
    def smooth_metrics(values: List[float], window_size: int = 10) -> List[float]:
        """Apply moving average smoothing to metrics."""
        if len(values) < window_size:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        
        return smoothed
    
    @staticmethod
    def calculate_derivative(values: List[float], steps: List[int]) -> List[float]:
        """Calculate the derivative (rate of change) of metrics."""
        if len(values) < 2:
            return [0.0] * len(values)
        
        derivatives = []
        for i in range(len(values)):
            if i == 0:
                # Forward difference for first point
                if len(values) > 1:
                    derivatives.append((values[1] - values[0]) / (steps[1] - steps[0]))
                else:
                    derivatives.append(0.0)
            elif i == len(values) - 1:
                # Backward difference for last point
                derivatives.append((values[-1] - values[-2]) / (steps[-1] - steps[-2]))
            else:
                # Central difference for middle points
                derivatives.append((values[i+1] - values[i-1]) / (steps[i+1] - steps[i-1]))
        
        return derivatives
    
    @staticmethod
    def find_local_minima(values: List[float]) -> List[int]:
        """Find indices of local minima in the values."""
        minima = []
        for i in range(1, len(values) - 1):
            if values[i] < values[i-1] and values[i] < values[i+1]:
                minima.append(i)
        return minima
    
    @staticmethod
    def compute_statistics(values: List[float]) -> Dict[str, float]:
        """Compute comprehensive statistics for a metric."""
        if not values:
            return {}
        
        values_np = np.array(values)
        return {
            'mean': float(np.mean(values_np)),
            'std': float(np.std(values_np)),
            'min': float(np.min(values_np)),
            'max': float(np.max(values_np)),
            'median': float(np.median(values_np)),
            'q25': float(np.percentile(values_np, 25)),
            'q75': float(np.percentile(values_np, 75)),
            'range': float(np.max(values_np) - np.min(values_np))
        }


class RunSelector:
    """Handle selection of training runs for analysis."""
    
    def __init__(self, logs_dir: str = "logs/raw_metrics"):
        self.logs_dir = Path(logs_dir)
        
    def list_available_runs(self) -> List[Dict[str, Any]]:
        """List all available training runs with metadata."""
        runs = []
        
        if not self.logs_dir.exists():
            return runs
        
        for run_dir in sorted(self.logs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
                
            run_info = {
                'name': run_dir.name,
                'path': str(run_dir),
                'created': None,
                'total_steps': 0,
                'has_config': False,
                'has_metrics': False
            }
            
            # Check for config file
            config_path = run_dir / "config.json"
            if config_path.exists():
                run_info['has_config'] = True
                run_info['created'] = datetime.fromtimestamp(config_path.stat().st_mtime)
            
            # Check for metrics file and count steps
            metrics_path = run_dir / "scalar_metrics.jsonl"
            if metrics_path.exists():
                run_info['has_metrics'] = True
                # Count lines to get approximate step count
                with open(metrics_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if 'step' in entry:
                                run_info['total_steps'] = max(run_info['total_steps'], entry['step'])
                        except:
                            continue
            
            runs.append(run_info)
        
        return runs
    
    def display_run_info(self, runs: List[Dict[str, Any]]):
        """Display formatted information about available runs."""
        if not runs:
            print("No training runs found in logs/raw_metrics/")
            return
        
        print("\n" + "="*80)
        print(" "*30 + "Available Training Runs")
        print("="*80)
        
        for i, run in enumerate(runs, 1):
            status = "✓" if run['has_metrics'] else "✗"
            print(f"\n{i}. [{status}] {run['name']}")
            
            if run['created']:
                print(f"   Created: {run['created'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            if run['total_steps'] > 0:
                print(f"   Total Steps: {run['total_steps']:,}")
            
            if run['has_config']:
                print(f"   Config: Available")
            
            if not run['has_metrics']:
                print(f"   ⚠️  No metrics found")
        
        print("\n" + "-"*80)
    
    def get_user_selection(self, runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get user's selection of run to analyze."""
        if not runs:
            return None
        
        while True:
            try:
                choice = input("\nSelect run number (or 'q' to quit): ").strip().lower()
                
                if choice == 'q':
                    return None
                
                idx = int(choice) - 1
                if 0 <= idx < len(runs):
                    selected = runs[idx]
                    if not selected['has_metrics']:
                        print("⚠️  This run has no metrics. Please select another.")
                        continue
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(runs)}")
            
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")


class TrainingAnalyzer:
    """Main analyzer for training dynamics and performance."""
    
    def __init__(self, run_path: str):
        self.run_path = Path(run_path)
        self.config = None
        self.metrics_data = defaultdict(list)
        self.steps = []
        
    def load_run_data(self) -> bool:
        """Load configuration and metrics for the run."""
        # Load config if available
        config_path = self.run_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Load metrics
        metrics_path = self.run_path / "scalar_metrics.jsonl"
        if not metrics_path.exists():
            print(f"Error: No metrics found at {metrics_path}")
            return False
        
        # Parse all metrics
        unique_steps = set()
        with open(metrics_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    metric_name = entry['name']
                    step = entry['step']
                    value = entry['value']
                    
                    self.metrics_data[metric_name].append((step, value))
                    unique_steps.add(step)
                except Exception:
                    continue
        
        self.steps = sorted(unique_steps)
        
        # Sort metrics by step
        for metric_name in self.metrics_data:
            self.metrics_data[metric_name].sort(key=lambda x: x[0])
        
        return len(self.metrics_data) > 0
    
    def detect_training_phases(self) -> List[TrainingPhase]:
        """Detect different phases in training (warmup, rapid descent, stabilization, etc.)."""
        phases = []
        
        if 'train/loss' not in self.metrics_data:
            return phases
        
        # Extract loss values and steps
        loss_data = self.metrics_data['train/loss']
        steps = [x[0] for x in loss_data]
        losses = [x[1] for x in loss_data]
        
        if len(losses) < 10:
            return phases
        
        # Smooth the loss curve
        smoothed_losses = MetricsProcessor.smooth_metrics(losses, window_size=10)
        
        # Calculate derivatives
        derivatives = MetricsProcessor.calculate_derivative(smoothed_losses, steps)
        
        # Detect phases based on derivative patterns
        
        # Phase 1: Warmup (first 5-10% of training)
        warmup_end = int(len(steps) * 0.1)
        if warmup_end > 0:
            warmup_losses = smoothed_losses[:warmup_end]
            phases.append(TrainingPhase(
                name="Warmup",
                start_step=steps[0],
                end_step=steps[min(warmup_end, len(steps)-1)],
                avg_loss=np.mean(warmup_losses),
                loss_reduction_rate=np.mean(derivatives[:warmup_end]),
                description="Initial learning rate warmup and model initialization"
            ))
        
        # Phase 2: Rapid Descent (steepest negative derivative)
        if len(derivatives) > warmup_end:
            # Find the period with most negative derivative
            window = min(50, len(derivatives) // 4)
            min_deriv_idx = warmup_end
            min_deriv_sum = float('inf')
            
            for i in range(warmup_end, len(derivatives) - window):
                window_sum = sum(derivatives[i:i+window])
                if window_sum < min_deriv_sum:
                    min_deriv_sum = window_sum
                    min_deriv_idx = i
            
            rapid_end = min(min_deriv_idx + window, len(steps) - 1)
            phases.append(TrainingPhase(
                name="Rapid Descent",
                start_step=steps[min_deriv_idx],
                end_step=steps[rapid_end],
                avg_loss=np.mean(smoothed_losses[min_deriv_idx:rapid_end]),
                loss_reduction_rate=np.mean(derivatives[min_deriv_idx:rapid_end]),
                description="Steepest loss reduction phase"
            ))
        
        # Phase 3: Stabilization (derivative approaches zero)
        stabilization_threshold = abs(np.percentile(derivatives, 10))  # Near-zero derivative
        stable_start = None
        
        for i in range(len(derivatives) // 2, len(derivatives)):
            if abs(derivatives[i]) < stabilization_threshold:
                stable_start = i
                break
        
        if stable_start:
            phases.append(TrainingPhase(
                name="Stabilization",
                start_step=steps[stable_start],
                end_step=steps[-1],
                avg_loss=np.mean(smoothed_losses[stable_start:]),
                loss_reduction_rate=np.mean(derivatives[stable_start:]),
                description="Loss stabilization and fine-tuning"
            ))
        
        return phases
    
    def analyze_loss_trends(self) -> Dict[str, Any]:
        """Analyze loss scale, trends, and inclination."""
        analysis = {
            'train': {},
            'validation': {}
        }
        
        # Analyze training loss
        if 'train/loss' in self.metrics_data:
            train_losses = [x[1] for x in self.metrics_data['train/loss']]
            train_steps = [x[0] for x in self.metrics_data['train/loss']]
            
            analysis['train'] = {
                'statistics': MetricsProcessor.compute_statistics(train_losses),
                'initial_loss': train_losses[0] if train_losses else None,
                'final_loss': train_losses[-1] if train_losses else None,
                'best_loss': min(train_losses) if train_losses else None,
                'best_step': train_steps[np.argmin(train_losses)] if train_losses else None,
                'total_reduction': (train_losses[0] - train_losses[-1]) if len(train_losses) > 1 else 0,
                'reduction_percentage': ((train_losses[0] - train_losses[-1]) / train_losses[0] * 100) if len(train_losses) > 1 and train_losses[0] > 0 else 0
            }
            
            # Calculate trend (linear regression slope)
            if len(train_losses) > 1:
                coeffs = np.polyfit(train_steps, train_losses, 1)
                analysis['train']['trend_slope'] = coeffs[0]
                analysis['train']['trend_description'] = "decreasing" if coeffs[0] < 0 else "increasing"
        
        # Analyze validation loss
        if 'validation/val_loss' in self.metrics_data:
            val_losses = [x[1] for x in self.metrics_data['validation/val_loss']]
            val_steps = [x[0] for x in self.metrics_data['validation/val_loss']]
            
            analysis['validation'] = {
                'statistics': MetricsProcessor.compute_statistics(val_losses),
                'initial_loss': val_losses[0] if val_losses else None,
                'final_loss': val_losses[-1] if val_losses else None,
                'best_loss': min(val_losses) if val_losses else None,
                'best_step': val_steps[np.argmin(val_losses)] if val_losses else None,
                'total_reduction': (val_losses[0] - val_losses[-1]) if len(val_losses) > 1 else 0,
                'reduction_percentage': ((val_losses[0] - val_losses[-1]) / val_losses[0] * 100) if len(val_losses) > 1 and val_losses[0] > 0 else 0
            }
            
            # Calculate trend
            if len(val_losses) > 1:
                coeffs = np.polyfit(val_steps, val_losses, 1)
                analysis['validation']['trend_slope'] = coeffs[0]
                analysis['validation']['trend_description'] = "decreasing" if coeffs[0] < 0 else "increasing"
            
            # Detect overfitting
            if len(train_losses) > 0 and len(val_losses) > 0:
                recent_train = np.mean(train_losses[-10:]) if len(train_losses) >= 10 else train_losses[-1]
                recent_val = np.mean(val_losses[-10:]) if len(val_losses) >= 10 else val_losses[-1]
                
                analysis['overfitting'] = {
                    'detected': recent_val > recent_train * 1.1,  # 10% threshold
                    'train_val_gap': recent_val - recent_train,
                    'gap_percentage': ((recent_val - recent_train) / recent_train * 100) if recent_train > 0 else 0
                }
        
        return analysis
    
    def detect_plateaus(self) -> List[PlateauInfo]:
        """Detect training plateaus where improvement stagnates."""
        plateaus = []
        
        if 'train/loss' not in self.metrics_data:
            return plateaus
        
        loss_data = self.metrics_data['train/loss']
        steps = [x[0] for x in loss_data]
        losses = [x[1] for x in loss_data]
        
        if len(losses) < 20:
            return plateaus
        
        # Smooth losses
        smoothed = MetricsProcessor.smooth_metrics(losses, window_size=10)
        
        # Define plateau detection parameters
        window_size = min(50, len(smoothed) // 4)
        improvement_threshold = 0.001  # Less than 0.1% improvement
        
        i = 0
        while i < len(smoothed) - window_size:
            window_losses = smoothed[i:i+window_size]
            window_improvement = (window_losses[0] - window_losses[-1]) / window_losses[0] if window_losses[0] > 0 else 0
            
            if abs(window_improvement) < improvement_threshold:
                # Found a plateau, extend it as far as possible
                plateau_start = i
                plateau_end = i + window_size
                
                # Extend plateau
                while plateau_end < len(smoothed) - 1:
                    next_improvement = (smoothed[plateau_start] - smoothed[plateau_end]) / smoothed[plateau_start] if smoothed[plateau_start] > 0 else 0
                    if abs(next_improvement) < improvement_threshold:
                        plateau_end += 1
                    else:
                        break
                
                duration = plateau_end - plateau_start
                
                # Classify severity
                if duration > len(smoothed) * 0.3:
                    severity = "severe"
                elif duration > len(smoothed) * 0.15:
                    severity = "moderate"
                else:
                    severity = "mild"
                
                plateaus.append(PlateauInfo(
                    start_step=steps[plateau_start],
                    end_step=steps[min(plateau_end, len(steps)-1)],
                    duration=duration,
                    avg_loss=np.mean(smoothed[plateau_start:plateau_end]),
                    improvement_rate=window_improvement,
                    severity=severity
                ))
                
                i = plateau_end
            else:
                i += 1
        
        return plateaus
    
    def calculate_improvements(self) -> Dict[str, Any]:
        """Calculate various improvement metrics."""
        improvements = {}
        
        # Perplexity improvements
        if 'train/perplexity' in self.metrics_data:
            perp_data = self.metrics_data['train/perplexity']
            perp_values = [x[1] for x in perp_data]
            
            if len(perp_values) > 1:
                improvements['perplexity'] = {
                    'initial': perp_values[0],
                    'final': perp_values[-1],
                    'best': min(perp_values),
                    'reduction': perp_values[0] - perp_values[-1],
                    'reduction_percentage': ((perp_values[0] - perp_values[-1]) / perp_values[0] * 100) if perp_values[0] > 0 else 0
                }
        
        # Learning rate schedule
        if 'train/learning_rate' in self.metrics_data:
            lr_data = self.metrics_data['train/learning_rate']
            lr_values = [x[1] for x in lr_data]
            
            improvements['learning_rate'] = {
                'initial': lr_values[0],
                'final': lr_values[-1],
                'max': max(lr_values),
                'min': min(lr_values),
                'schedule_type': self._detect_lr_schedule(lr_values)
            }
        
        # Gradient norm stability
        if 'train/grad_norm' in self.metrics_data:
            grad_data = self.metrics_data['train/grad_norm']
            grad_values = [x[1] for x in grad_data]
            
            improvements['gradient_stability'] = {
                'mean': np.mean(grad_values),
                'std': np.std(grad_values),
                'max': max(grad_values),
                'min': min(grad_values),
                'exploding_risk': any(g > 100 for g in grad_values),
                'vanishing_risk': any(g < 0.0001 for g in grad_values)
            }
        
        # Best checkpoint info
        if 'validation/val_loss' in self.metrics_data:
            val_data = self.metrics_data['validation/val_loss']
            val_losses = [x[1] for x in val_data]
            val_steps = [x[0] for x in val_data]
            
            if val_losses:
                best_idx = np.argmin(val_losses)
                improvements['best_checkpoint'] = {
                    'step': val_steps[best_idx],
                    'val_loss': val_losses[best_idx],
                    'position': f"{(best_idx / len(val_losses) * 100):.1f}% through training"
                }
        
        return improvements
    
    def _detect_lr_schedule(self, lr_values: List[float]) -> str:
        """Detect the type of learning rate schedule used."""
        if len(lr_values) < 10:
            return "unknown"
        
        # Check if constant
        if np.std(lr_values) < 1e-10:
            return "constant"
        
        # Check if monotonically decreasing
        is_decreasing = all(lr_values[i] >= lr_values[i+1] for i in range(len(lr_values)-1))
        
        # Check for warmup
        has_warmup = any(lr_values[i] < lr_values[i+1] for i in range(min(20, len(lr_values)-1)))
        
        if has_warmup and is_decreasing:
            return "cosine_with_warmup"
        elif is_decreasing:
            return "cosine_annealing"
        else:
            return "custom"
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            'run_name': self.run_path.name,
            'phases': [phase.__dict__ for phase in self.detect_training_phases()],
            'loss_trends': self.analyze_loss_trends(),
            'plateaus': [plateau.__dict__ for plateau in self.detect_plateaus()],
            'improvements': self.calculate_improvements(),
            'config': self.config
        }
        
        # Add summary statistics
        if self.steps:
            report['summary'] = {
                'total_steps': self.steps[-1],
                'metrics_tracked': list(self.metrics_data.keys()),
                'has_validation': 'validation/val_loss' in self.metrics_data,
                'data_points': sum(len(v) for v in self.metrics_data.values())
            }
        
        return report


class ReportGenerator:
    """Generate formatted reports from analysis results."""
    
    @staticmethod
    def format_console_output(report: Dict[str, Any]):
        """Format analysis report for console display."""
        print("\n" + "="*80)
        print(" "*25 + f"Training Analysis: {report['run_name']}")
        print("="*80)
        
        # Summary
        if 'summary' in report:
            print("\n📊 SUMMARY")
            print("-"*40)
            print(f"Total Steps: {report['summary']['total_steps']:,}")
            print(f"Metrics Tracked: {', '.join(report['summary']['metrics_tracked'])}")
            print(f"Has Validation: {'Yes' if report['summary']['has_validation'] else 'No'}")
            print(f"Total Data Points: {report['summary']['data_points']:,}")
        
        # Training Phases
        if report['phases']:
            print("\n🔄 TRAINING PHASES")
            print("-"*40)
            for phase in report['phases']:
                print(f"\n{phase['name']} (Steps {phase['start_step']:,} - {phase['end_step']:,})")
                print(f"  • Average Loss: {phase['avg_loss']:.4f}")
                print(f"  • Loss Reduction Rate: {phase['loss_reduction_rate']:.6f}")
                print(f"  • {phase['description']}")
        
        # Loss Analysis
        if 'loss_trends' in report:
            print("\n📉 LOSS ANALYSIS")
            print("-"*40)
            
            # Training loss
            if 'train' in report['loss_trends'] and report['loss_trends']['train']:
                train = report['loss_trends']['train']
                print("\nTraining Loss:")
                print(f"  • Initial: {train.get('initial_loss', 'N/A'):.4f}")
                print(f"  • Final: {train.get('final_loss', 'N/A'):.4f}")
                print(f"  • Best: {train.get('best_loss', 'N/A'):.4f} (Step {train.get('best_step', 'N/A')})")
                print(f"  • Total Reduction: {train.get('reduction_percentage', 0):.2f}%")
                
                if 'statistics' in train:
                    stats = train['statistics']
                    print(f"  • Mean±Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                    print(f"  • Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
            # Validation loss
            if 'validation' in report['loss_trends'] and report['loss_trends']['validation']:
                val = report['loss_trends']['validation']
                print("\nValidation Loss:")
                print(f"  • Initial: {val.get('initial_loss', 'N/A'):.4f}")
                print(f"  • Final: {val.get('final_loss', 'N/A'):.4f}")
                print(f"  • Best: {val.get('best_loss', 'N/A'):.4f} (Step {val.get('best_step', 'N/A')})")
                print(f"  • Total Reduction: {val.get('reduction_percentage', 0):.2f}%")
            
            # Overfitting detection
            if 'overfitting' in report['loss_trends']:
                overfit = report['loss_trends']['overfitting']
                if overfit['detected']:
                    print(f"\n⚠️  Overfitting Detected!")
                    print(f"  • Train-Val Gap: {overfit['train_val_gap']:.4f} ({overfit['gap_percentage']:.2f}%)")
                else:
                    print(f"\n✅ No significant overfitting detected")
        
        # Plateaus
        if report['plateaus']:
            print("\n⚠️  TRAINING PLATEAUS")
            print("-"*40)
            for i, plateau in enumerate(report['plateaus'], 1):
                print(f"\nPlateau {i} ({plateau['severity'].upper()})")
                print(f"  • Steps: {plateau['start_step']:,} - {plateau['end_step']:,}")
                print(f"  • Duration: {plateau['duration']} steps")
                print(f"  • Average Loss: {plateau['avg_loss']:.4f}")
                print(f"  • Improvement Rate: {plateau['improvement_rate']:.6f}")
        else:
            print("\n✅ No significant plateaus detected")
        
        # Improvements
        if 'improvements' in report:
            print("\n📈 IMPROVEMENTS & METRICS")
            print("-"*40)
            
            # Perplexity
            if 'perplexity' in report['improvements']:
                perp = report['improvements']['perplexity']
                print(f"\nPerplexity:")
                print(f"  • Initial: {perp['initial']:.2f}")
                print(f"  • Final: {perp['final']:.2f}")
                print(f"  • Best: {perp['best']:.2f}")
                print(f"  • Reduction: {perp['reduction_percentage']:.2f}%")
            
            # Learning rate
            if 'learning_rate' in report['improvements']:
                lr = report['improvements']['learning_rate']
                print(f"\nLearning Rate:")
                print(f"  • Schedule: {lr['schedule_type']}")
                print(f"  • Initial: {lr['initial']:.2e}")
                print(f"  • Final: {lr['final']:.2e}")
            
            # Gradient stability
            if 'gradient_stability' in report['improvements']:
                grad = report['improvements']['gradient_stability']
                print(f"\nGradient Stability:")
                print(f"  • Mean Norm: {grad['mean']:.4f}")
                print(f"  • Std Dev: {grad['std']:.4f}")
                
                if grad['exploding_risk']:
                    print(f"  • ⚠️  Risk of exploding gradients detected")
                if grad['vanishing_risk']:
                    print(f"  • ⚠️  Risk of vanishing gradients detected")
            
            # Best checkpoint
            if 'best_checkpoint' in report['improvements']:
                best = report['improvements']['best_checkpoint']
                print(f"\nBest Checkpoint:")
                print(f"  • Step: {best['step']:,}")
                print(f"  • Validation Loss: {best['val_loss']:.4f}")
                print(f"  • Position: {best['position']}")
        
        print("\n" + "="*80)
    
    @staticmethod
    def export_analysis(report: Dict[str, Any], output_path: str):
        """Export analysis report to JSON file."""
        output_path = Path(output_path)
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Analysis exported to: {output_path}")
    
    @staticmethod
    def create_comparison_table(reports: List[Dict[str, Any]]):
        """Create comparison table for multiple runs."""
        if not reports:
            return
        
        print("\n" + "="*80)
        print(" "*30 + "Run Comparison")
        print("="*80)
        
        # Header
        print(f"\n{'Run Name':<30} {'Steps':<10} {'Final Loss':<12} {'Best Val':<12} {'Reduction':<10}")
        print("-"*80)
        
        for report in reports:
            name = report['run_name'][:28]
            steps = report.get('summary', {}).get('total_steps', 0)
            
            # Get loss values
            train_loss = report.get('loss_trends', {}).get('train', {})
            val_loss = report.get('loss_trends', {}).get('validation', {})
            
            final_loss = train_loss.get('final_loss', float('nan'))
            best_val = val_loss.get('best_loss', float('nan'))
            reduction = train_loss.get('reduction_percentage', 0)
            
            print(f"{name:<30} {steps:<10,} {final_loss:<12.4f} {best_val:<12.4f} {reduction:<10.2f}%")
        
        print("-"*80)


def main():
    """Main entry point for the analysis tool."""
    parser = argparse.ArgumentParser(description="Analyze training logs for LLM models")
    parser.add_argument('--run', type=str, help='Specific run name to analyze')
    parser.add_argument('--compare', nargs='+', help='Compare multiple runs')
    parser.add_argument('--export', type=str, help='Export analysis to JSON file')
    parser.add_argument('--logs-dir', type=str, default='logs/raw_metrics',
                      help='Directory containing training logs')
    
    args = parser.parse_args()
    
    # Run selector
    selector = RunSelector(args.logs_dir)
    
    if args.compare:
        # Compare multiple runs
        reports = []
        for run_name in args.compare:
            run_path = Path(args.logs_dir) / run_name
            if run_path.exists():
                analyzer = TrainingAnalyzer(str(run_path))
                if analyzer.load_run_data():
                    reports.append(analyzer.generate_report())
                else:
                    print(f"Failed to load data for run: {run_name}")
        
        if reports:
            ReportGenerator.create_comparison_table(reports)
    
    elif args.run:
        # Analyze specific run
        run_path = Path(args.logs_dir) / args.run
        if not run_path.exists():
            print(f"Error: Run '{args.run}' not found in {args.logs_dir}")
            sys.exit(1)
        
        analyzer = TrainingAnalyzer(str(run_path))
        if analyzer.load_run_data():
            report = analyzer.generate_report()
            ReportGenerator.format_console_output(report)
            
            if args.export:
                ReportGenerator.export_analysis(report, args.export)
        else:
            print(f"Failed to load data for run: {args.run}")
    
    else:
        # Interactive mode
        runs = selector.list_available_runs()
        
        if not runs:
            print("No training runs found. Please train a model first.")
            sys.exit(0)
        
        selector.display_run_info(runs)
        selected = selector.get_user_selection(runs)
        
        if selected:
            print(f"\nAnalyzing: {selected['name']}...")
            
            analyzer = TrainingAnalyzer(selected['path'])
            if analyzer.load_run_data():
                report = analyzer.generate_report()
                ReportGenerator.format_console_output(report)
                
                # Ask if user wants to export
                export = input("\nExport analysis to JSON? (y/n): ").strip().lower()
                if export == 'y':
                    output_name = f"analysis_{selected['name']}.json"
                    ReportGenerator.export_analysis(report, output_name)
            else:
                print("Failed to load run data")


if __name__ == "__main__":
    main()