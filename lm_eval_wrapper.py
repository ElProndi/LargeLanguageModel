#!/usr/bin/env python3
"""
LM Evaluation Harness Wrapper for TransformerLM

This module provides a wrapper to interface our custom TransformerLM model
with EleutherAI's lm-evaluation-harness framework for benchmarking.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, Iterator
import numpy as np

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from model import TransformerLM
from tokenizer import WikipediaTokenizer


@register_model("transformer_lm")
class TransformerLMWrapper(LM):
    """
    Wrapper class for TransformerLM to work with lm-evaluation-harness.
    
    This implements the required interface for perplexity and generation evaluations.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 512,
        **kwargs
    ):
        """
        Initialize the wrapper with a model checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Cache tokenizer special tokens
        self.bos_token_id = self.tokenizer.tokenizer.token_to_id("<BOS>")
        self.eos_token_id = self.tokenizer.tokenizer.token_to_id("<EOS>")
        self.pad_token_id = self.tokenizer.tokenizer.token_to_id("<PAD>")
        
        # Set model vocab size
        self.vocab_size = self.model.vocab_size
        
    def _load_model_and_tokenizer(self) -> Tuple[TransformerLM, WikipediaTokenizer]:
        """Load model from checkpoint and tokenizer."""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint or default file
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            with open("config.json", 'r') as f:
                config = json.load(f)
        
        # Create model
        model = TransformerLM(
            vocab_size=config['model']['vocab_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            max_position_embeddings=config['model']['max_position_embeddings'],
            dropout=0.0,  # Disable dropout for evaluation
            attention_dropout=0.0,
            layer_norm_eps=config['model']['layer_norm_eps'],
            initializer_range=config['model']['initializer_range'],
            use_cache=config['model']['use_cache'],
            pad_token_id=config['tokenizer'].get('pad_token_id', 3)
        )
        
        # Handle compiled model state dict
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        model.eval()
        
        # Load tokenizer
        tokenizer = WikipediaTokenizer()
        tokenizer_paths = [
            Path("tokenizers/full_tokenizer"),
            Path("tokenizers/test_tokenizer")
        ]
        
        for tokenizer_path in tokenizer_paths:
            if tokenizer_path.exists():
                try:
                    tokenizer.load(str(tokenizer_path))
                    break
                except Exception:
                    continue
        
        return model, tokenizer
    
    @property
    def eot_token_id(self) -> int:
        """End of text token ID (same as EOS for our model)."""
        return self.eos_token_id
    
    @property
    def max_gen_toks(self) -> int:
        """Maximum generation length."""
        return 256
    
    @property
    def batch_size(self) -> int:
        """Batch size for evaluation."""
        return self._batch_size
    
    @property
    def device(self) -> torch.device:
        """Device the model is on."""
        return self.device
    
    def tok_encode(self, string: str) -> List[int]:
        """Encode a string to token IDs."""
        tokens = self.tokenizer.encode(string)
        # Add BOS token if not present
        if len(tokens) == 0 or tokens[0] != self.bos_token_id:
            tokens = [self.bos_token_id] + tokens
        return tokens
    
    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(tokens)
    
    def loglikelihood(
        self, 
        requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of completions given contexts.
        
        Args:
            requests: List of (context, completion) pairs
            
        Returns:
            List of (log-likelihood, is_greedy) tuples
        """
        results = []
        
        # Process in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = self._loglikelihood_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _loglikelihood_batch(
        self,
        batch: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """Process a batch of loglikelihood requests."""
        results = []
        
        with torch.no_grad():
            for context, completion in batch:
                # Tokenize context and completion
                context_tokens = self.tok_encode(context) if context else [self.bos_token_id]
                completion_tokens = self.tok_encode(completion)
                
                # Remove BOS from completion if present
                if completion_tokens and completion_tokens[0] == self.bos_token_id:
                    completion_tokens = completion_tokens[1:]
                
                # Combine tokens
                all_tokens = context_tokens + completion_tokens
                
                # Truncate if necessary
                if len(all_tokens) > self.max_length:
                    # Keep context and as much completion as possible
                    all_tokens = all_tokens[:self.max_length]
                    completion_tokens = all_tokens[len(context_tokens):]
                
                # Convert to tensor
                input_ids = torch.tensor([all_tokens], dtype=torch.long, device=self.device)
                
                # Get model output
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                # Calculate log probabilities for completion tokens
                # We need to align: logits[i] predicts token[i+1]
                start_idx = len(context_tokens) - 1  # Start from last context token
                end_idx = len(all_tokens) - 1
                
                if start_idx < 0 or end_idx <= start_idx:
                    # Edge case: no valid completion to score
                    results.append((float('-inf'), False))
                    continue
                
                # Get relevant logits and targets
                completion_logits = logits[0, start_idx:end_idx]  # [comp_len, vocab]
                completion_targets = input_ids[0, start_idx+1:end_idx+1]  # [comp_len]
                
                # Calculate log probabilities
                log_probs = F.log_softmax(completion_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    dim=-1, 
                    index=completion_targets.unsqueeze(-1)
                ).squeeze(-1)
                
                # Sum log probabilities
                total_log_prob = token_log_probs.sum().item()
                
                # Check if greedy (if argmax matches actual tokens)
                greedy_tokens = completion_logits.argmax(dim=-1)
                is_greedy = torch.equal(greedy_tokens, completion_targets)
                
                results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(
        self,
        requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """
        Compute rolling log-likelihood for perplexity evaluation.
        
        This method processes text with a sliding window approach.
        """
        results = []
        
        for request in requests:
            # For rolling, we only have the text (no separate context/completion)
            text = request[0] if isinstance(request, tuple) else request
            
            # Tokenize
            tokens = self.tok_encode(text)
            
            # Process with sliding window if text is long
            if len(tokens) <= self.max_length:
                # Single forward pass
                result = self._compute_rolling_loglikelihood(tokens)
                results.append(result)
            else:
                # Use sliding window with stride
                stride = self.max_length // 2
                total_log_prob = 0.0
                total_tokens = 0
                
                for start_idx in range(0, len(tokens) - 1, stride):
                    end_idx = min(start_idx + self.max_length, len(tokens))
                    window_tokens = tokens[start_idx:end_idx]
                    
                    log_prob, _ = self._compute_rolling_loglikelihood(window_tokens)
                    
                    # Weight by number of predictions in this window
                    num_predictions = len(window_tokens) - 1
                    if num_predictions > 0:
                        total_log_prob += log_prob
                        total_tokens += num_predictions
                    
                    if end_idx >= len(tokens):
                        break
                
                # Average log probability
                avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else float('-inf')
                results.append((avg_log_prob, False))
        
        return results
    
    def _compute_rolling_loglikelihood(
        self,
        tokens: List[int]
    ) -> Tuple[float, bool]:
        """Compute log-likelihood for a sequence of tokens."""
        if len(tokens) < 2:
            return (float('-inf'), False)
        
        with torch.no_grad():
            # Convert to tensor
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            
            # Get model output
            outputs = self.model(input_ids)
            logits = outputs['logits']
            
            # Calculate log probabilities
            # logits[0, i] predicts tokens[i+1]
            pred_logits = logits[0, :-1]  # [seq_len-1, vocab]
            target_tokens = input_ids[0, 1:]  # [seq_len-1]
            
            # Get log probabilities
            log_probs = F.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1,
                index=target_tokens.unsqueeze(-1)
            ).squeeze(-1)
            
            # Sum log probabilities
            total_log_prob = token_log_probs.sum().item()
            
            # Check if greedy
            greedy_tokens = pred_logits.argmax(dim=-1)
            is_greedy = torch.equal(greedy_tokens, target_tokens)
            
        return (total_log_prob, is_greedy)
    
    def generate_until(
        self,
        requests: List[dict]
    ) -> List[str]:
        """
        Generate text until a stop condition is met.
        
        Args:
            requests: List of generation requests with 'context' and 'until' keys
            
        Returns:
            List of generated strings
        """
        results = []
        
        for request in requests:
            context = request['context']
            until = request.get('until', [])
            max_gen_toks = request.get('max_gen_toks', self.max_gen_toks)
            temperature = request.get('temperature', 0.0)
            
            # Tokenize context
            input_ids = self.tok_encode(context)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_tensor,
                    max_length=len(input_ids) + max_gen_toks,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_k=1 if temperature == 0 else 50,
                    top_p=1.0 if temperature == 0 else 0.95,
                    eos_token_id=self.eos_token_id
                )
            
            # Extract generated tokens (excluding input)
            generated_tokens = generated_ids[0, len(input_ids):].tolist()
            
            # Decode to text
            generated_text = self.tok_decode(generated_tokens)
            
            # Truncate at stop strings if provided
            if until:
                for stop_str in until:
                    if stop_str in generated_text:
                        generated_text = generated_text[:generated_text.index(stop_str)]
                        break
            
            results.append(generated_text)
        
        return results