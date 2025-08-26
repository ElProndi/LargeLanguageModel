# CLAUDE.md - LLM Training Pipeline

This file provides guidance to Claude Code when working with the LLM training pipeline implementation.

## Project Overview

A from-scratch implementation of a transformer-based language model training pipeline, designed for efficiency and clarity. The pipeline processes Wikipedia data to train a custom language model using modern best practices.

## Directory Structure

```
LargeLanguageModel/
├── CLAUDE.md              # This file - project guidance
├── config.json            # Centralized configuration for entire pipeline
├── cleaner.py             # Wikipedia data extraction and cleaning
├── tokenizer.py           # BPE tokenizer training with Hugging Face
├── dataset_prep.py        # Dataset tokenization with sliding windows
├── dataloader.py          # PyTorch DataLoader for efficient training
├── model.py               # Transformer language model architecture
├── train.py               # Main training loop with mixed precision
├── Inference.py           # Interactive text generation from checkpoints
├── utils/                 # Training utilities
│   ├── __init__.py        # Package initialization
│   ├── logging_utils.py   # Dual logging (TensorBoard + JSON)
│   ├── metrics.py         # Metrics tracking and smoothing
│   └── scheduler.py       # Learning rate scheduling
├── tokenizers/            # Saved tokenizer models
│   ├── test_tokenizer/    # Test mode tokenizer (256 vocab)
│   └── full_tokenizer/    # Full tokenizer (16384 vocab)
├── checkpoints/           # Model checkpoints
│   ├── checkpoint_latest.pt    # Most recent checkpoint
│   ├── checkpoint_best.pt      # Best validation loss
│   └── checkpoint_step_*.pt    # Step-based checkpoints
└── logs/                  # Training logs
    ├── tensorboard/       # TensorBoard event files
    └── raw_metrics/       # Raw JSON metrics for analysis
```

## Data Organization

```
Desktop/
├── data/                    # Centralized training data
│   ├── raw/                # Raw Wikipedia JSONL files
│   │   └── enwiki_namespace_0/
│   │       └── *.jsonl     # 38 Wikipedia dump files (50GB+)
│   ├── cleaned_articles/   # Processed ASCII text
│   │   └── cleaned_*.txt   # One article per line format
│   └── tokenized_datasets/ # Tokenized sequences for training
│       ├── test_dataset/   # Test mode output
│       │   ├── tokens_*.npy    # NumPy arrays of token sequences
│       │   └── metadata.json   # Dataset statistics
│       └── full_dataset/   # Full dataset output
│           ├── tokens_*.npy    # NumPy arrays (38 files)
│           └── metadata.json   # Overall statistics
└── LargeLanguageModel/     # Active development directory
```

## Pipeline Architecture

### Phase 1: Data Cleaning (cleaner.py)
**Purpose**: Extract and clean Wikipedia articles from raw JSONL dumps

**Key Features**:
- Processes 38 Wikipedia JSONL files sequentially
- Extracts article text from nested JSON structure
- Cleans text to pure ASCII for consistent tokenization
- Removes metadata, references, and formatting artifacts
- Implements efficient streaming processing with minimal memory footprint

**Input**: Raw Wikipedia dumps at `/home/andrea/Desktop/data/raw/enwiki_namespace_0/*.jsonl`
**Output**: Cleaned text at `/home/andrea/Desktop/data/cleaned_articles/cleaned_*.txt`

**Processing Characteristics**:
- Multi Processor, each process 1 file
- Real-time statistics: articles processed, processing speed, file progress
- Output format: One complete article per line

### Phase 2: Tokenization (tokenizer.py)
**Purpose**: Train custom BPE tokenizer on Wikipedia corpus for optimal compression

**Key Features**:
- ByteLevel BPE tokenization preserving all ASCII characters
- Configurable vocabulary size (16384 for full, 256 for test)
- Special tokens for sequence boundaries (<BOS>, <EOS>, <UNK>)
- Memory-efficient streaming from cleaned articles
- Fast batch encoding/decoding with parallel processing

**Input**: Cleaned articles from `/home/andrea/Desktop/data/cleaned_articles/`
**Output**: Saved tokenizer models in `tokenizers/` directory

**Processing Characteristics**:
- Progress bars during vocabulary learning
- Perfect reconstruction of ASCII text
- Compression ratio: ~3-4x on typical English text
- Automatic save/load of trained models

### Phase 3: Dataset Preparation (dataset_prep.py)
**Purpose**: Tokenize cleaned articles into training sequences with sliding windows

**Key Features**:
- Loads trained BPE tokenizer (test or full mode)
- Processes cleaned files sequentially with memory efficiency
- Creates sliding windows with 50% overlap for continuity
- Batch tokenization using Hugging Face multi-threading
- Saves sequences as NumPy uint16 arrays for efficient loading

**Input**: Cleaned articles from `/home/andrea/Desktop/data/cleaned_articles/`
**Output**: Tokenized sequences in `/home/andrea/Desktop/data/tokenized_datasets/`

**Processing Characteristics**:
- Test mode: First 10,000 articles only for quick iteration
- Configurable window size (default 512 tokens)
- Batch processing (default 10,000 articles per batch)
- Real-time progress tracking with speed statistics
- Metadata generation with compression metrics

### Phase 4: Data Loading (dataloader.py)
**Purpose**: High-performance PyTorch DataLoader for training with full dataset in RAM/GPU

**Key Features**:
- Loads entire tokenized dataset into RAM at initialization (~2GB)
- Optional GPU memory loading for zero-copy training
- Implements standard PyTorch Dataset interface
- Automatic concatenation of all tokenized files
- Memory usage reporting and statistics

**Input**: Tokenized sequences from `/home/andrea/Desktop/data/tokenized_datasets/full_dataset/`
**Output**: PyTorch DataLoader ready for training

**Processing Characteristics**:
- Fails fast if dataset is incomplete (missing metadata.json)
- Direct tensor indexing with no I/O during training
- Automatic optimization when data is on GPU (num_workers=0)
- Configurable batch size, shuffling, and worker processes
- Persistent workers for CPU loading scenarios

### Phase 5: Model Architecture (model.py)
**Purpose**: Transformer-based language model using native PyTorch modules

**Key Features**:
- Uses PyTorch's native `nn.TransformerDecoder` for efficiency
- Learned positional embeddings (GPT-2 style)
- Tied input/output embeddings to reduce parameters
- Support for autoregressive text generation
- Configurable architecture via config.json

**Generation Capabilities**:
- Temperature-based sampling
- Top-k and Top-p (nucleus) sampling
- Batch generation with proper EOS handling
- Support for padding and attention masks

### Phase 6: Model Training (train.py)
**Purpose**: Complete training pipeline with GPU optimization and monitoring

**Key Features**:
- Gradient accumulation for effective batch size scaling
- Cosine learning rate scheduling with linear warmup
- Dual logging system (TensorBoard + raw JSON)
- Automatic checkpointing with interruption recovery
- Memory-efficient train/val split with GPU/CPU optimization

**Architecture**:
- GPU-resident training data for zero-copy access
- CPU-resident validation data with pinned memory
- Non-blocking async transfers for validation
- Windowed metrics smoothing for stable monitoring
- Early stopping and best model tracking

**Training Characteristics**:
- Requires CUDA GPU (fails fast if not available)
- Validation every 10% of epoch
- Automatic checkpoint management
- Graceful interruption handling with state preservation

### Phase 7: Utilities (utils/)
**Purpose**: Modular components for training infrastructure

**Components**:
1. **DualLogger** (logging_utils.py):
   - Simultaneous TensorBoard and JSON logging
   - Experiment tracking with timestamps
   - Buffered writes for efficiency
   - Configuration and hyperparameter logging

2. **MetricsTracker** (metrics.py):
   - Sliding window smoothing for noisy metrics
   - Throughput calculation (tokens/second)
   - Memory usage tracking
   - Best metrics persistence
   - Early stopping support

3. **Scheduler** (scheduler.py):
   - Cosine annealing with warmup
   - Configurable minimum learning rate
   - Resume-friendly state management
   - Integration with PyTorch optimizers

### Phase 8: Model Inference (Inference.py)
**Purpose**: Interactive text generation using trained model checkpoints

**Key Features**:
- Interactive checkpoint selection with file details
- Two input modes: custom prompts or predefined examples
- Configurable generation parameters (temperature, top-k, top-p)
- Color-coded terminal interface with ANSI codes
- Real-time generation statistics and performance metrics
- Continuous generation loop without model reloading

**Input**: Trained checkpoints from `checkpoints/` directory
**Output**: Generated text with statistics

**Processing Characteristics**:
- Automatic GPU/CPU detection and optimization
- Handles compiled model state dictionaries
- Falls back between full and test tokenizers
- Memory-efficient with GPU cache management
- Batch generation support with proper EOS handling

**Generation Parameters**:
- **Max Length**: Maximum tokens to generate (default: 200)
- **Temperature**: Controls randomness (default: 0.8)
- **Top-k**: Limits vocabulary to top k tokens (default: 50)
- **Top-p**: Nucleus sampling threshold (default: 0.95)

### Upcoming Phases (To Be Implemented)

**Phase 9: Inference Server**
- REST API for text generation
- Streaming generation support
- Model quantization for deployment

## Usage

### Running the Data Cleaner
```bash
# Test mode, test 50000 articles per process.
python3 cleaner.py --test

# Processes all 38 Wikipedia files with 4 workers
python3 cleaner.py

# Output: /home/andrea/Desktop/data/cleaned_articles/
```

### Training and Using the Tokenizer
```bash
# Test mode: small vocab (256) on first cleaned file
python3 tokenizer.py --test

# Full training: 16384 vocab on all cleaned files
python3 tokenizer.py

# Encode text with trained tokenizer
python3 tokenizer.py --encode "Hello world"

# Decode token IDs
python3 tokenizer.py --decode "123,456,789"

# Load existing tokenizer
python3 tokenizer.py --load tokenizers/full_tokenizer

# Output: tokenizers/test_tokenizer/ or tokenizers/full_tokenizer/
```

### Preparing the Dataset
```bash
# Test mode: tokenize first 10,000 articles with test tokenizer
python3 dataset_prep.py --test

# Full dataset: tokenize all 38 cleaned files
python3 dataset_prep.py

# Custom window size (default 512)
python3 dataset_prep.py --window 1024

# Custom batch size (default 10000)
python3 dataset_prep.py --batch 5000

# Output: /home/andrea/Desktop/data/tokenized_datasets/
```

### Creating and Testing the Model
```python
# Create model from configuration
from model import create_model

model = create_model("config.json")
# Output: Model with ~25M parameters

# Test forward pass
import torch
input_ids = torch.randint(0, 16384, (batch_size, seq_len))
outputs = model(input_ids, labels=input_ids)
print(f"Loss: {outputs['loss'].item()}")

# Generate text
generated = model.generate(
    prompt_ids,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
```

### Using the DataLoader
```python
# In your training script
from dataloader import create_simple_train_val_dataloaders

# Create memory-efficient train/val dataloaders
train_loader, val_loader, info = create_simple_train_val_dataloaders(
    batch_size=64,
    val_split=0.1,      # 10% validation split
    shuffle_train=True,
    test_mode=False,    # Set True for quick testing
    verbose=True,
    seed=42            # Consistent split across runs
)

# Training loop
for epoch in range(num_epochs):
    # Training data is on GPU for zero-copy access
    for batch in train_loader:
        # batch shape: (batch_size, window_size)
        outputs = model(batch)  # Already on GPU
        loss = criterion(outputs)
        # ... training step
    
    # Validation data is on CPU with pinned memory
    for batch in val_loader:
        batch = batch.to('cuda', non_blocking=True)
        # ... validation step

# Access dataset info
print(f"Train sequences: {info['train_size']:,}")
print(f"Val sequences: {info['val_size']:,}")
print(f"Train device: {info['train_device']}")
print(f"Val device: {info['val_device']}")
```

### Training the Model
```bash
# Test mode: single epoch on small dataset
python3 train.py --test

# Full training: complete dataset with all epochs
python3 train.py

# Resume from checkpoint
python3 train.py --resume checkpoints/checkpoint_latest.pt

# Custom experiment name for logging
python3 train.py --name experiment_lr_1e3

# Monitor training with TensorBoard
tensorboard --logdir logs/tensorboard

# Training outputs:
# - Checkpoints in checkpoints/
# - TensorBoard logs in logs/tensorboard/
# - Raw metrics in logs/raw_metrics/
```

### Running Inference
```bash
# Run interactive inference with trained model
python3 Inference.py

# The script will:
# 1. List available checkpoints with details
# 2. Load selected model and tokenizer
# 3. Offer custom or predefined prompts
# 4. Generate text with configurable parameters
# 5. Display generation statistics

# Example workflow:
# - Select checkpoint_best.pt for best model
# - Choose "Write custom prompt" mode
# - Enter "The future of AI is"
# - Use default generation parameters
# - View generated text and statistics
# - Continue generating or exit
```

## Configuration System

The `config.json` file centralizes all configuration for the pipeline:

## Development Guidelines

### Code Philosophy
- **Clarity over cleverness**: Readable, maintainable code
- **Modular design**: Single responsibility per module
- **Efficiency at scale**: Handle 50GB+ datasets gracefully
- **Fail fast**: Clear error messages, no silent failures

### Implementation Standards
- Implement proper error handling with descriptive messages
- Provide progress feedback for long-running operations
- Memory-efficient processing for large datasets
- Clean separation of concerns between modules
- Keep main classes focused on their core functionality
- Extract complex operations to dedicated modules

## Error Handling

The pipeline implements strict error handling:
- Immediate failure on critical errors
- Clear error messages with context
- No automatic fallbacks that hide problems
- Progress state preserved on interruption

## Next Steps
1. Extend inference capabilities
   - REST API for text generation
   - Streaming generation support
   - Web-based interface
   - Model serving optimizations

2. Optimization and scaling
   - Model quantization for deployment
   - Multi-GPU training support
   - Distributed data parallel training
   - Flash attention integration

---

*This document will be updated as new components are implemented.*